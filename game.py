## make it so then people can add custom games to be trained here
### 1: needs the type of neural net and how the game state is inputted
### 2: needs a play method which will play the game between the current and given player (or other method of 1v1)
### 3: needs a advanced version to strive towards (optional hunter killer such as minimax to encourage strong growth)

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

############

from player import Player

from player import Connect_4_Player
from player import Connect_4_Human

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
        raise NotImplementedError

    def make_player(self):
        """Makes a player suitable for this game"""
        raise NotImplementedError

    def load_players(self, dir_path):
        """Creates players by loading existing neural nets"""


class Connect_4(Game):

    player = Connect_4_Player

    rows = 7
    columns = 6

    def play(self, player_one, player_two, verbose = 0):
        # Make an empty board
        board = np.array([np.array([0.0 for i in range(self.rows)]) for j in range(self.columns)]).reshape(1, self.columns, self.rows, 1)

        game_over = False
        winner = None
        current_player = player_one
        other_player = player_two

        ##watching = True
        ##if random.uniform(0.0, 1.0) == 1.0:
            ##watching = True

        # Keep playing the game till there is a winner
        while(not game_over):
            best_move = current_player.pick_move(board)

            valid_move_check, board = self.make_move(board, best_move)

            if not valid_move_check:
                return other_player
            
            if verbose == 1:
                self.visualize_board(board[0], current_player == player_one)


            #     if is_player_one:
            #          print("-P1:", best_move)
            #     else:
            #          print("-P2", best_move)

            ##print(valid_move_check)

            game_over, win_num = self.check_game_over(board[0])

            if win_num == 1:
                winner = current_player
            if win_num == -1:
                winner = other_player

            if current_player == player_one:
                current_player = player_two
                other_player = player_one
            else:
                current_player = player_one
                other_player = player_two

            self.flip_board_perspective(board[0])

        # If it was a draw play again (need to adjust this prevent long looping)
        if winner == None:
            winner = self.play(player_two, player_one, self.columns, self.rows)

        return winner

    # Flips the board perspective as the nets are trained so that they are the '1s'
    def flip_board_perspective(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j][0] != 0.0:
                    board[i][j][0] *= -1.0


    # Checks to see if the game is over or not
    def check_game_over(self, board):
        col_num = len(board)
        for i in range(col_num):
            row_num = len(board[i])
            for j in range(row_num):
                cur_symbol = board[i][j][0]
                if cur_symbol == 0.0 or (i + 3 >= col_num and j + 3 >= row_num):
                    continue
                
                four_in_a_row = True
                if i + 3 < col_num:
                    for k in range(1, 4):
                        if board[i + k][j][0] != cur_symbol:
                            four_in_a_row = False
                            break
                    if four_in_a_row:
                        return True, cur_symbol

                four_in_a_row = True
                if j + 3 < row_num:
                    for k in range(1, 4):
                        if board[i][j + k][0] != cur_symbol:
                            four_in_a_row = False
                            break
                    if four_in_a_row:
                        return True, cur_symbol

                four_in_a_row = True
                if j + 3 < row_num and i + 3 < col_num:
                    for k in range(1, 4):
                        if board[i + k][j + k][0] != cur_symbol:
                            four_in_a_row = False
                            break
                    if four_in_a_row:
                        return True, cur_symbol

        # See if the board is full and there is a tie
        board_full = True
        for i in board:
            if 0.0 in i:
                board_full = False
                break

        return board_full, None

    # Makes a move and returns the changed board as well as if it was valid or not
    def make_move(self, board, move_column, other = False):
        for i in range(len(board[0][move_column])):
            if board[0][move_column][i][0] == 0.0:
                if not other:
                    board[0][move_column][i][0] = 1.0
                else:
                    board[0][move_column][i][0] = -1.0
                return True, board

        return False, board

    def evaluate(self, p):
        return self.play(p, Connect_4_Human(), 1)

    def play_versus(self):
        return self.play(Connect_4_Human(), Connect_4_Human())

    def make_player(self, net = None):
        if not net is None:
            return Connect_4_Player(net)

        # Set hyper params
        filter_dim = 2
        drop_prob = 0.5

        conv_dim = 64
        dense_dim = 128

        num_conv_layers = 2
        num_dense_layers = 1

        model = Sequential()

        # Add conv layers
        for i in range(num_conv_layers):
            model.add(layers.Conv2D(conv_dim, (filter_dim, filter_dim), activation = 'relu', input_shape=(self.columns, self.rows, 1)))
            model.add(layers.BatchNormalization()) 
            model.add(layers.Dropout(drop_prob)) 

        model.add(layers.Flatten())

        # Add fully connected layers
        for i in range(num_dense_layers):
            model.add(layers.Dense(dense_dim, activation='relu'))

        model.add(layers.Dense(self.columns, activation = 'softmax'))

        return Connect_4_Player(model)

    # Print out the board so humans can look at it
    def visualize_board(self, board, player_one):
        

        print_str = ""

        for i in range(1, self.rows + 1):
            inv_row = self.rows - i
            for col in range(self.columns):
                char = '*'
                if board[col][inv_row][0] == 1.0:
                    if player_one:
                        char = 'X'
                    else:
                        char = 'O'
                elif board[col][inv_row][0] == -1.0:
                    if not player_one:
                        char = 'X'
                    else:
                        char = 'O'
                    
                print_str += char + '  '

            print_str += '\n'

        print(print_str)
        