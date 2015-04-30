#!/usr/bin/env python
from __future__ import print_function

import logging
import re
import sys
import gomill
from gomill import common, boards, ascii_boards, handicap_layout, sgf, sgf_moves
#import gomill.boards
#import gomill.ascii_boards
#import gomill.handicap_layout

def slimmify(txt):
    lines = txt.replace('  ', ' ').split('\n')
    lines[-1] = ' ' + lines[-1]
    return '\n'.join(lines)
    
def gtp_io():
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
    opponent_passed = False
    ko_forbidden = None

    while True:
        try:
            line = raw_input()
        except EOFError:
            break
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'#.*', '', line)
        cmdline = line.strip().split()
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
        elif cmd == "komi":
            komi = float(args[0])
        elif cmd == "play":
            m = gomill.common.move_from_vertex(args[1], boardsize)
            if m is not None:
                row, col = m
                ko_forbidden = board.play(row, col, args[0])
            else:
                pass
                opponent_passed = True
        elif cmd == "genmove":
            ## TODO smarter pass
            if opponent_passed:
                ret = 'pass'
            else:
                ## TODO smarter play
                ret = 'a1'
        elif cmd == "reg_genmove":
            ## TODO
            ret = None
            err = 'not implemented'
        elif cmd == "loadsgf":
            filename, movenum_limit = args[0], 100000
            # optional movenum_limit argument
            if len(args) > 1:
                movenum_limit = int(args[1])
            try:
                with open(filename, 'r') as fin:
                    game = gomill.sgf.Sgf_game.from_string(fin.read())
                board, movepairs = gomill.sgf_moves.get_setup_and_moves(game)
                for move_num, (color, (row, col)) in zip(xrange(movenum_limit), movepairs):
                    ko_forbidden = board.play(row, col, color)
            except IOError:
                err = 'cannot open sgf file'
                ret = None
        # Tournament commands
        elif cmd in ['fixed_handicap', 'place_free_handicap']:
            handicaps = gomill.handicap_layout.handicap_points(int(args[0]), boardsize)
            board.apply_setup(handicaps, [], [])
            ret = ' '.join(map(gomill.common.format_vertex, handicaps))
        elif cmd == 'set_free_handicap':
            for pt in args:
                row, col = gomill.common.move_from_vertex(pt, boardsize)
                ko_forbidden = board.play(row, col, 'b')
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
            logging.debug("\n" + slimmify(gomill.ascii_boards.render_board(board)))
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret), end='')
        else:
            print('?%s %s\n\n' % (cmdid, err), end='')
        sys.stdout.flush()
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    gtp_io()
    

