


class Player:
    """A player for a game containing the network and methods to pick moves"""

    def pick_move(board):
        """Pick a move to make and return it"""


class Connect_4_Player(Player):
    """A player for connect 4"""

    net = None
    gen = 0

    def __init__(self, net, gen) -> None:
        self.net = net
        self.generation = gen

    def pick_move(board):
        return 0

class Connect_4_Human(Player):
    """A human for connect 4"""

    def pick_move(board):
        return int(input("What Col?"))