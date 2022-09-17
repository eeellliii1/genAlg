import numpy as np


class Player:
    """A player for a game containing the network and methods to pick moves"""

    def pick_move(selfboard):
        """Pick a move to make and return it"""
        raise NotImplementedError

    def save(self):
        """Save the network used by this player"""
        raise NotImplementedError


class Connect_4_Player(Player):
    """A player for connect 4"""

    net = None

    def __init__(self, net) -> None:
        self.net = net

    def pick_move(self, board):
        options = self.net.predict(board)[0]

        best_columns = []
        best_chance = -float('inf')
        for i in range(len(options)):
            if options[i] > best_chance:
                best_columns = [i]
                best_chance = options[i]
            elif options[i] >= best_chance:
                best_columns.append(i)

        best_move = np.random.choice(best_columns)

        return best_move

    def save(self, filepath):
        self.net.save(filepath)

class Connect_4_Human(Player):
    """A human for connect 4"""

    def pick_move(board):
        return int(input("What Col?"))

    