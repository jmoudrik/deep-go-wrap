from __future__ import print_function

import logging
import numpy as np
import tempfile
import copy

import gomill
from gomill import common, boards, sgf, sgf_moves, gtp_states

import utils
import analyze_board

"""
Basic Player / Bot objects;

Player should be gomill compatible envelope which actually generates
    moves, resigns, ..

Bot should be the object that actually does the core work, e.g. computing
Move probability, ..
"""

class Player(object):
    def __init__(self):
        self.handlers = { 'name' : self.handle_name,
                          'quit' : self.handle_quit }
        self.name = None
    def genmove(self, game_state, player):
        """
        game_state is gomill.gtp_states.Game_state

        :returns: gomill.Move_generator_result
        """
        raise NotImplementedError
    def handle_name(self, args):
        if self.name is None:
            return self.__class__.__name__
        return self.name
    def handle_quit(self, args):
        pass
    def get_handlers(self):
        return self.handlers
    def __str__(self):
        return "<%s>"%self.handle_name([])

class DistWrappingMaxPlayer(Player):
    """
    A simple wrapping bot which chooses next move to be the one with the biggest (therefore the name)
    probability. The probabilities are computed by the wrapped bot's gen_probdist().
    """
    def __init__(self, bot):
        super(DistWrappingMaxPlayer,  self).__init__()
        self.bot = bot
        self.handlers['ex-dist'] = self.handle_ex_dist
        self.handlers['move_probabilities'] = self.handle_move_probabilities
        self.move_num = 0
    def genmove(self, game_state, player):
        self.move_num += 1
        dist = self.bot.gen_probdist(game_state, player)
        result = gtp_states.Move_generator_result()
        if dist is not None:
            move = np.unravel_index(np.argmax(dist), dist.shape)
            result.move = move
            logging.debug("%s valid moves\n%s"%(self,
                                                utils.dist_stats(dist)))
            logging.debug("%s move %d: playing %s"%(self,
                                                    self.move_num,
                                                    gomill.common.format_vertex(move)))
        else:
            result.pass_move = True
            logging.debug("%s move %d: playing pass"%(self, self.move_num))
        return result
    def handle_quit(self, args):
        self.bot.close()
    def handle_move_probabilities(self, args):
        return self.bot.move_probabilities()
    def handle_ex_dist(self, args):
        top = 3
        if args:
            try:
                top = gomill.gtp_engine.interpret_int(args[0])
            except IndexError:
                gtp_engine.report_bad_arguments()

        return self.bot.dist_stats(top)


class DistWrappingSamplingPlayer(Player):
    """
    A simple wrapping bot which randomly samples next move based on the moves' probability
    distribution, computed by the wrapped bot's gen_probdist().

    Never passes.
    """
    def __init__(self, bot):
        super(DistWrappingSamplingPlayer,  self).__init__()
        self.bot = bot
    def genmove(self, game_state, player):
        dist = self.bot.gen_probdist(game_state, player)
        result = gtp_states.Move_generator_result()
        if dist is not None:
            # choose an intersection with probability given by the dist
            coord = np.random.choice((game_state.board.side ** 2), 1, p=dist.ravel())[0]
            move = (coord / game_state.board.side,  coord % game_state.board.side)
            result.move = move
        else:
            result.pass_move = True
        return result
    def handle_quit(self, args):
        self.bot.close()


class RandomPlayer(Player):
    def genmove(self, game_state, player):
        result = gtp_states.Move_generator_result()
        # pass
        if game_state.move_history and not game_state.move_history[-1].move:
            result.pass_move = True
            return result
        else:
            for i in xrange(10):
                row, col = np.random.choice(game_state.board.side, 2)
                ## TODO this might be incorrect move
                # but nobody will use the RandomPlayer anyway
                if not game_state.board.get(row, col):
                    result.move =  (row, col)
                    return result
            result.resign = True
            return result

class WrappingGnuGoPlayer(Player):
    def __init__(self, player, passing=True, resigning=False):
        super(WrappingGnuGoPlayer,  self).__init__()
        self.player = player
        self.passing = passing
        self.resigning = resigning

        hp = copy.copy(player.get_handlers())
        hp.update(self.handlers)
        self.handlers = hp

    def genmove(self, game_state, color):
        result = gtp_states.Move_generator_result()

        logging.debug("%s enter"%(self))
        move = self.gnu_go_move(game_state, color)
        # pass if GnuGo tells us to do so
        if self.passing and move == 'pass':
            result.pass_move = True
            return result
        elif self.resigning and move == 'resign':
            result.resign = True
            return result
        else:
            logging.debug("%s not listening, descend"%(self))
            return self.player.genmove(game_state, color)

    def gnu_go_move(self, game_state, color):
        assert isinstance(game_state.board, gomill.boards.Board) # for wingide code completion

        game = gomill.sgf.Sgf_game(size=game_state.board.side)
        gomill.sgf_moves.set_initial_position(game, game_state.board)
        node = game.get_root()
        node.set('KM', game_state.komi)
        node.set('PL', color)

        with tempfile.NamedTemporaryFile() as sgf_file:
            sgf_file.write(game.serialise())
            sgf_file.flush()
            gg_move = utils.get_gnu_go_response(sgf_file.name, color)

        return gg_move.lower()


class DistributionBot(object):
    def __init__(self):
        self.last_dist = None
        self.last_player = None
    def __str__(self):
        return "<%s>"%(self.__class__.__name__)
    def gen_probdist_raw(self, game_state, player):
        """
        The core method to implement for distribution bots.
        It needs not

        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
                 the array should be normalized to 1
        """
        raise NotImplementedError
    def gen_probdist(self, game_state, player):
        """
        Generates a correct move probability distribution for the next move,
        using the gen_probdist_raw().

        Correct means that it zeroes out probability of playing incorrect move,
        such as move forbidden by ko, suicide and occupied points.

        Stores the dist and the player.

        :return: a numpy array of floats of shape (board.side, board.side), or None for pass
                 the array is normalized to 1
        """
        dist = self.gen_probdist_raw(game_state, player)

        if dist is not None:
            correct_moves = analyze_board.board2correct_move_mask(game_state.board,  player)
            if game_state.ko_point:
                correct_moves[game_state.ko_point[0]][game_state.ko_point[1]] = 0

            # compute some debugging stats of the incorrect moves first
            incorrect_dist = (1 - correct_moves) * dist
            logging.debug("%s incorrect moves\n%s"%(self,
                                utils.dist_stats(incorrect_dist)))

            # keep only correct moves
            dist = correct_moves * dist
            s = dist.sum()
            if s > 0.0:
                dist = dist / dist.sum()
            else:
                logging.debug("No valid moves, PASSING.")
                dist = None

        self.last_dist = dist
        self.last_player = player
        return self.last_dist

    def move_probabilities(self):
        if self.last_dist is not None:
            ret = []
            for row, col in np.transpose(np.nonzero(self.last_dist)):
                ret.append( "%s %f"%(gomill.common.format_vertex((row, col)),
                                     self.last_dist[row][col]))
            return '\n'.join(ret)
        return ''

    def dist_stats(self, top=3):
        if self.last_dist is not None:
            return utils.dist_stats(self.last_dist, top)
        return ''

    def close(self):
        """Called upon exit, to allow for resource freeup."""
        pass


class RandomDistBot(DistributionBot):
    def gen_probdist_raw(self, game_state, player):
        return np.random.random((game_state.board.side, game_state.board.side))


if __name__ == "__main__":
    def test_bot():
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
        player = DistWrappingMaxPlayer(RandomDistBot())

        class State:
            pass
        s = State()

        b = gomill.boards.Board(3)
        s.board = b
        b.play(1, 1, "b")
        b.play(0, 1, "b")
        logging.debug("\n"+gomill.ascii_boards.render_board(b))
        mv = player.genmove(s, 'w').move
        b.play(mv[0], mv[1], 'w')
        logging.debug("\n"+gomill.ascii_boards.render_board(b))
        logging.debug("best move is " + gomill.common.format_vertex(mv))
        logging.debug("\n" + str(player.bot.last_dist))
        logging.debug(utils.dist_stats(player.bot.last_dist))

    test_bot()

