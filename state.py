import gomill
from gomill import common, boards, ascii_boards, handicap_layout, sgf, sgf_moves

def slimmify(txt):
    # The default gomill board printing is bloated with spaces
    lines = txt.replace('  ', ' ').split('\n')
    lines[-1] = ' ' + lines[-1]
    return '\n'.join(lines)

def format_move(move):
    if move in [None, 'pass']:
        return str(move)
    return gomill.common.format_vertex(move)
    
class State:
    def __init__(self, board, last_move=None, ko_forbidden_move=None, komi=0.0):
        # gomill board
        self.board = board
        
        # last_move = None indicates game has not started
        # last_move = 'pass'
        # last_move = (3,3), ...
        self.last_move = last_move
        
        # move (x,y) forbidden by ko, or None
        self.ko_forbidden_move = ko_forbidden_move
        
        self.komi = komi
    def __str__(self):
        return """
%s
last_move = %s
ko_forbidden_move = %s
komi = %.1f"""%(slimmify(gomill.ascii_boards.render_board(self.board)),
                format_move(self.last_move), 
                format_move(self.ko_forbidden_move), 
                self.komi)
