## make it so then people can add custom games to be trained here
### 1: needs the type of neural net and how the game state is inputted
### 2: needs a play method which will play the game between the current and given player (or other method of 1v1)
### 3: needs a advanced version to strive towards (optional hunter killer such as minimax to encourage strong growth)

from player import Player

class Game:
    """Interface for game"""

    # The input dimension for the neural net as a tuple
    input_dim = None;

    def play(self, p1, p2):
        """Plays two networks against eachother, returning the victorious one"""
        raise NotImplementedError

    def evaluate(self, p):
        """Allows the user to evaluate a network by playing against it"""
        raise NotImplementedError

    def play_versus(self):
        """Lets two humans play the game against eachother"""


class Connect_4(Game):

    def play(self, p1, p2):
        return p1

    def evaluate(self, p):
        return p

    def play_versus(self):
        return 0