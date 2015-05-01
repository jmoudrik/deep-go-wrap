#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys
import gomill
from gomill import common, boards, ascii_boards, handicap_layout, sgf, sgf_moves

import bots

def slimmify(txt):
    lines = txt.replace('  ', ' ').split('\n')
    lines[-1] = ' ' + lines[-1]
    return '\n'.join(lines)

def format_move(move):
    if move in [None, 'pass']:
        return str(move)
    return gomill.common.format_vertex(move)
    
def gtp_io(bot):
    """GTP protocol I/O
    
    Inspired by michi, https://github.com/pasky/michi
    Spec as in http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    """
    known_commands = [ # required commands
                      'protocol_version', 'name', 'version', 'known_command',
                      'list_commands', 'quit', 'boardsize', 'clear_board',
                      'komi', 'play', 'genmove', 
                       # tournament subset
                      'fixed_handicap', 'place_free_handicap', 'set_free_handicap', 
                       # regression
                      'loadsgf', 'reg_genmove',
                       # extensions
                      'ex-info' ]
    
    boardsize = 19
    komi = 6.5
    board = None
    # last_move = None indicates game has not started
    # last_move = 'pass'
    # last_move = (3,3), ...
    last_move = None
    next_color = None
    # move (x,y) forbidden by ko, or None
    ko_forbidden_move = None

    while True:
        try:
            line = raw_input()
        except EOFError:
            break
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'#.*', '', line)
        cmdline = line.lower().strip().split()
        if not cmdline:
            continue
        logging.debug("got gtp cmd: " + ' '.join(cmdline))
        
        cmdid = ''
        if re.match('\d+', cmdline[0]):
            cmdid = cmdline[0]
            cmdline = cmdline[1:]
            
        cmd, args = cmdline[0], cmdline[1:]
        
        ret, err = '', '???'
        # Core commands
        if cmd == "boardsize":
            boardsize = int(args[0])
            board = None
        elif cmd == "clear_board":
            board = gomill.boards.Board(boardsize)
            next_color, last_move, ko_forbidden_move = 'b', None, None
        elif cmd == "komi":
            komi = float(args[0])
        elif cmd == "play":
            color, move = args[0], gomill.common.move_from_vertex(args[1], boardsize)
            if next_color and next_color != color:
                logging.warn("This move's color='%s', but I expected it to be next_color='%s'!"%(color, next_color))
            if move is not None:
                row, col = move
                ko_forbidden_move = board.play(row, col, color)
                last_move, next_color = move, gomill.common.opponent_of(color)
            else:
                last_move, next_color = 'pass', gomill.common.opponent_of(color)
        elif cmd in ["reg_genmove",  "genmove"]:
            color = args[0]
            if next_color and next_color != color:
                logging.warn("This move's color='%s', but I expected it to be next_color='%s'!"%(color, next_color))
                
            # the difference between genmove and reg_genmove is that
            # reg_genmove does not update the board
            move = bot.genmove(board, color, last_move, ko_forbidden_move, komi)
            if move:
                ret = gomill.common.format_vertex(move)
            else:
                ret = 'pass'
                
            if cmd == 'genmove':
                next_color = gomill.common.opponent_of(color)
                if move:
                    row, col = move
                    board.play(row, col, color)
        elif cmd == "loadsgf":
            filename, movenum_limit = args[0], 100000
            # optional movenum_limit argument
            if len(args) > 1:
                movenum_limit = int(args[1])
            try:
                with open(filename, 'r') as fin:
                    game = gomill.sgf.Sgf_game.from_string(fin.read())
                
                board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
                komi = game.get_komi()
                # we wanna have correct state even in the case of empty game
                ko_forbidden_move, last_move, next_color = None, None, None
                for move_num, (color, move) in zip(xrange(movenum_limit), movepairs):
                    if move:
                        row, col = move
                        ko_forbidden_move = board.play(row, col, color)
                        last_move, next_color = move, gomill.common.opponent_of(color)
                    else:
                        last_move = 'pass'
            except IOError:
                err = 'cannot open sgf file'
                ret = None
        # Tournament commands
        elif cmd in ['fixed_handicap', 'place_free_handicap']:
            ## TODO how to initialize last_move?
            if int(args[0]):
                handicaps = gomill.handicap_layout.handicap_points(int(args[0]), boardsize)
                board.apply_setup(handicaps, [], [])
                ret = ' '.join(map(gomill.common.format_vertex, handicaps))
            next_color = 'w'
        elif cmd == 'set_free_handicap':
            if args:
                for pt in args:
                    row, col = gomill.common.move_from_vertex(pt, boardsize)
                    ko_forbidden_move = board.play(row, col, 'b')
            next_color = 'w'
        # Extensions
        elif cmd == "ex-info":
            ## TODO
            ret = None
            err = 'not implemented'
        # Misc
        elif cmd == "name":
            ret = 'deepgowrap'
        elif cmd == "version":
            ret = '0.1, all your neural nets are belong to us'
        elif cmd == "list_commands":
            ret = '\n'.join(known_commands)
        elif cmd == "known_command":
            ret = 'true' if args[0] in known_commands else 'false'
        elif cmd == "protocol_version":
            ret = '2'
        elif cmd == "quit":
            print('=%s \n\n' % (cmdid,), end='')
            break
        else:
            logging.warn('Ignoring unknown command: "%s"' % (line))
            ret = None
            err = 'unknown command "%s"' % cmd
            
        if board:
            logging.debug("""
%s
last_move = %s
next_color = %s
ko_forbidden_move = %s
komi = %.1f"""%(slimmify(gomill.ascii_boards.render_board(board)),
                format_move(last_move), 
                next_color,
                format_move(ko_forbidden_move), 
                komi))
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret), end='')
        else:
            print('?%s %s\n\n' % (cmdid, err), end='')
        sys.stdout.flush()
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    bot = bots.RandomBot()
    gtp_io(bot)
    

