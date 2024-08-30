##TODO make it so then people can add custom games to be trained here
### 1: needs the type of neural net and how the game state is inputted
### 2: needs a play method which will play the game between the current and given player (or other method of 1v1)
### 3: needs a advanced version to strive towards (optional hunter killer such as minimax to encourage strong growth)

import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

############

from player import Player

from player import Connect_4_Player
from player import Connect_4_Human

# Generic game running framework
class Game:
    """Interface for game"""

    # Display name string for menu
    name = "Connect 4"

    # The input dimension for the neural net as a tuple
    input_dim = None

    # Plays two networks against eachother, returning the victorious one
    def play(self, p1, p2):
        raise NotImplementedError

    # Allows the user to evaluate a network by playing against it
    def evaluate(self, p):
        raise NotImplementedError

    # Lets two humans play the game against eachother
    def play_versus(self):
        raise NotImplementedError

    # Makes a player suitable for this game
    def make_player(self):
        raise NotImplementedError

    # Creates players by loading existing neural nets
    def load_players(self, dir_path):
        raise NotImplementedError

##########################################

# Methods to run a game of Connect 4
class Connect_4(Game):

    # Define what type of player is used 
    player = Connect_4_Player

    # Connect 4 board size
    rows = 7
    columns = 6

    # The main method to play a game of connect 4
    #   verbose parameter determines if the board is displayed in console or not
    #       0 (default) is no
    #       1 is yes 
    def play(self, player_one, player_two, verbose = 0):
        
        # Make an empty board
        board = np.array([np.array([0.0 for i in range(self.rows)]) for j in range(self.columns)]).reshape(1, self.columns, self.rows, 1)

        # Set up game parameters
        game_over = False
        winner = None
        current_player = player_one
        other_player = player_two

        # Keep playing the game till there is a winner
        while(not game_over):
            # Get a move from the current player
            move = current_player.pick_move(board)

            # Check if the move is valid and make the move
            valid_move_check, board = self.make_move(board, move)

            # If the move is not valid, the other player wins by default
            if not valid_move_check:
                return other_player
            
            # If verbose is enabled, display the board in the console
            if verbose == 1:
                self.visualize_board(board[0], current_player == player_one)


            #     if is_player_one:
            #          print("-P1:", best_move)
            #     else:
            #          print("-P2", best_move)

            ##print(valid_move_check)

            # Check to see if the game is over, and if so determine a winner
            game_over, win_num = self.check_game_over(board[0])
            
            if win_num == 1:
                winner = current_player
            if win_num == -1:
                winner = other_player

            # Switch active player
            if current_player == player_one:
                current_player = player_two
                other_player = player_one
            else:
                current_player = player_one
                other_player = player_two

            # Flip board display 
            self.flip_board_perspective(board[0])

        # If it was a draw select the winner at random
        if winner == None:
            winner = np.random.choice([player_one, player_two])

        # Return the winner
        return winner

    # Flips the board perspective so the current player is always the 1s
    def flip_board_perspective(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j][0] != 0.0:
                    board[i][j][0] *= -1.0

    # Checks each tile to see if there is 4 in a row
    def check_game_over(self, board):
        col_num = len(board)
        for i in range(col_num):
            row_num = len(board[i])
            for j in range(row_num):
                cur_symbol = 1 #board[i][j][0]
                if board[i][j] != cur_symbol or (i + 3 >= col_num and j + 3 >= row_num):
                    continue
                
                # Check horizontally
                four_in_a_row = True
                if i + 3 < col_num:
                    for k in range(1, 4):
                        if board[i + k][j][0] != cur_symbol:
                            four_in_a_row = False
                            break
                    if four_in_a_row:
                        return True, cur_symbol

                # Check vertically
                four_in_a_row = True
                if j + 3 < row_num:
                    for k in range(1, 4):
                        if board[i][j + k][0] != cur_symbol:
                            four_in_a_row = False
                            break
                    if four_in_a_row:
                        return True, cur_symbol

                # Check diagonally
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
        
        # Return the game status
        return board_full, None

    # Makes a move and returns the changed board as well as if it was valid or not
    def make_move(self, board, move_column):
        for i in range(len(board[0][move_column])):

            # Check to see if the move is valid
            if board[0][move_column][i][0] == 0.0:
                
                # Make the move and return that it was valid
                board[0][move_column][i][0] = 1.0
                return True, board
        
        # Return the move was not valid
        return False, board

    # Make a match between a human and nerual net player
    def evaluate(self, p):

        # Determine who goes first
        players = [p, Connect_4_Human()]
        p1_index = random.randint(0, 2) 

        print(p1_index)

        return self.play(players[p1_index], players[1 - p1_index], 1)

    # Make a match between two human players
    def play_versus(self):
        return self.play(Connect_4_Human(), Connect_4_Human(), 1)

    # Make a neural net player
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
        
        # Make a string to show the display 
        print_str = ""

        # Build the board display string
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

        # Output the display string to the console
        print(print_str)
        
##########################################