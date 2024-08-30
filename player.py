import numpy as np

# The template for 
class Player:
    # Determine a game action and return it
    def pick_move(selfboard):
        raise NotImplementedError

    # Save the network used by this player
    def save(self):
        raise NotImplementedError
    
##########################################

# A neural net-based player for connect 4
class Connect_4_Player(Player): #TODO make this where neural network is specified so diff networks could compete in same game. Careful of cross-breeding issues however
    # Start with no neural net
    net = None

    # Can be given a net to start with
    def __init__(self, net) -> None:
        self.net = net

    # Given the current board state, query neural net for the percieved best move
    def pick_move(self, board):
        options = self.net.predict(board, verbose = 0)[0]

        # Store one or more best option
        best_columns = []

        # Find highest/tied option
        best_chance = -float('inf')
        for i in range(len(options)):
            if options[i] > best_chance:
                best_columns = [i]
                best_chance = options[i]
            elif options[i] >= best_chance:
                best_columns.append(i)

        # Determine the best move or a random one from among tied options
        best_move = np.random.choice(best_columns)

        return best_move

    #Saves the neural net at the given filepath as .h5 file
    def save(self, filepath):
        self.net.save(filepath)

# Allows a human user to play connect 4
class Connect_4_Human(Player):

    def pick_move(self, board):
        return int(input("What Col?"))

##########################################