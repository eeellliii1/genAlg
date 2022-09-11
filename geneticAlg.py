from time import sleep
import numpy as np
import os
import random
import sys
import heapq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# TODO 
# 1. Individual weight adjustment
# 2. 'Terminator' Minimax competitors
# 3. Variable population size
# 4. Test on boards of dimensions other than training?


# Splice net weights together and generate new generations
def reproduce(parents, columns, rows):
    model = make_model(columns, rows) # NEED TO ELIMINATE THIS just edit weights of the existing models

    # Pick out 2 parents (may adjust later to just take from anything in the parent pool)
    pars = []
    if len(parents) == 2:
        pars = [parents[0], parents[1]] # If there are only 2 parents take those
    else:
        pars = np.random.choice(parents, 2) # Take 2 random nets to be parents

    # Mutate value for multiplying weights by mutate_val -> 1 + mutate_val
    mutate_val = 0.5

    # Get the parent model layers
    layers = model.layers
    p1_layers = pars[0].layers
    p2_layers = pars[1].layers

    # Mix up the weights
    for i in range(len(layers)):
        layer = layers[i]
        p1_layer = p1_layers[i]
        p2_layer = p2_layers[i]

        layer_weights = layer.get_weights()
        p1_layer_weights = p1_layer.get_weights()
        p2_layer_weights = p2_layer.get_weights()
        
        for j in range(len(layer_weights)):
            layer_weights[j]
            p1_layer_weights[j]
            p2_layer_weights[j]

            choice = np.random.choice([0, 1, 2])
            if choice == 0:
                layer_weights[j] = p1_layer_weights[j]
            elif choice == 1:
                layer_weights[j] = p2_layer_weights[j]
            else:
                layer_weights[j] = (p1_layer_weights[j] + p2_layer_weights[j]) / 2

            if random.uniform(0, 1) >= 0.5:
                mut_v = random.uniform(mutate_val, mutate_val + 1)
                layer_weights[j] *= mut_v

        # Set the weights
        layer.set_weights(layer_weights)

    return model


# Make a neural net with default weights
def make_model(columns, rows):

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
        model.add(layers.Conv2D(conv_dim, (filter_dim, filter_dim), activation = 'relu', input_shape=(columns, rows, 1)))
        model.add(layers.BatchNormalization()) 
        model.add(layers.Dropout(drop_prob)) 

    model.add(layers.Flatten())

    # Add fully connected layers
    for i in range(num_dense_layers):
        model.add(layers.Dense(dense_dim, activation='relu'))

    model.add(layers.Dense(columns, activation = 'softmax'))

    return model


# Load all the models in the given file_path to a folder and return list of keras models
def load_models(file_path):
    file_names = os.listdir(file_path)

    models = []
    for file_name in file_names:
        models.append(tf.keras.models.load_model(file_path + "/" + file_name))

    return models


# Pit random pairs of models against eachother returning only the ones who win
def tournament(players, columns, rows):
    round = 1

    players_left = [i for i in range(len(players))] # Keep track of indices of who has not fought yet

    winners = []

    # Play matches until there is nobody left
    while len(players_left) >= 2:
        p1_index = np.random.choice(players_left)
        p1 = players[p1_index]
        players_left.remove(p1_index)

        p2_index = np.random.choice(players_left)
        p2 = players[p2_index]
        players_left.remove(p2_index)

        winners.append(play(p1, p2, columns, rows))

        round += 1

    return winners


# Play a game of two nets against eachother
def play(player_one, player_two, columns, rows):

    # Make an empty board
    board = np.array([np.array([0.0 for i in range(rows)]) for j in range(columns)]).reshape(1, columns, rows, 1)

    game_over = False
    winner = None
    current_player = player_one
    other_player = player_two

    ##watching = True
    ##if random.uniform(0.0, 1.0) == 1.0:
        ##watching = True

    # Keep playing the game till there is a winner
    while(not game_over):
        is_player_one = True
        if current_player != player_one:
            is_player_one = False

        options = current_player.predict(board)[0]
        ##if is_player_one:
            ##print(options)
        ##print(options)

        best_columns = []
        best_chance = -float('inf')
        for i in range(len(options)):
            if options[i] > best_chance:
                best_columns = [i]
                best_chance = options[i]
            elif options[i] >= best_chance:
                best_columns.append(i)

        best_move = np.random.choice(best_columns)

        valid_move_check, board = make_move(board, best_move)

        if not valid_move_check:
            return other_player
        
        # if watching:
        #     visualize_board(board[0], columns, rows, player_one)


        #     if is_player_one:
        #          print("-P1:", best_move)
        #     else:
        #          print("-P2", best_move)

        ##print(valid_move_check)

        game_over, win_num = check_game_over(board[0])

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

        flip_board_perspective(board[0])

    # If it was a draw play again (need to adjust this prevent long looping)
    if winner == None:
        winner = play(player_two, player_one, columns, rows)

    return winner


# Makes a move and returns the changed board as well as if it was valid or not
def make_move(board, move_column, other = False):
    for i in range(len(board[0][move_column])):
        if board[0][move_column][i][0] == 0.0:
            if not other:
                board[0][move_column][i][0] = 1.0
            else:
                board[0][move_column][i][0] = -1.0
            return True, board

    return False, board


# Checks to see if the game is over or not
def check_game_over(board):
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


# Flips the board perspective as the nets are trained so that they are the '1s'
def flip_board_perspective(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j][0] != 0.0:
                board[i][j][0] *= -1.0


# Print out the board so humans can look at it
def visualize_board(board, columns, rows):
    print_str = ""

    for i in range(1, rows + 1):
        inv_row = rows - i
        for col in range(columns):
            char = '*'
            if board[col][inv_row][0] == 1.0:
                char = 'X'
            elif board[col][inv_row][0] == -1.0:
                    char = 'O'
                
            print_str += char + '  '

        print_str += '\n'

    print(print_str)

#########################################

args = sys.argv[1:]
# Command Struct 1: train {new/old} {load_dir} {save_dir} {generations} {(not needed if old) num_players}
#   only setup right now with generation size divisible by 4
# Command Struct 2: evaluate {load_dir}
#   will assign each net in the dir a number between 0 and the total - 1 in order of how they are listed in the dir
# Command Struct 3: best {load_dir} {top_how_many}
#   finds the net with the best win rate against all other nets

rows = 7
columns = 6

# Trains nets either new or loaded
if args[0] == "train":
    num_players = 0 # Number of nets per generation
    generations = int(args[4]) 

    # Make brand new nets
    if args[1] == "new":
        num_players = int(args[5])
        players = [make_model(columns, rows) for i in range(num_players)]
    # Load previously generated nets
    else:
        players = load_models(args[2])
        num_players = len(players)

    # Train the nets for {generations} generations
    for i in range(generations):
        print("Generation:", i + 1)

        # Play through two halvings of the population (needs changing so input does not need to be a multiple of 4)
        winners = tournament(players, columns, rows)
        winners_winners = tournament(winners, columns, rows)

        players = [reproduce(winners_winners, columns, rows) for i in range(num_players)]

        # Every x generations save the models
        x = 60
        if (i + 1) % x == 0:
            for i in range(len(players)):
                players[i].save(args[3] + '/net' + str(i) + '.h5')

    for i in range(len(players)):
        players[i].save(args[3] + '/net' + str(i) + '.h5')

# Lets you play against the model of your choice
elif args[0] == "evaluate":
    players = load_models(args[1])

    usr_input = ""
    while(usr_input != "exit"):

        usr_input = input("Choose a challenger: 0 ->" + str(len(players) - 1))
        if usr_input == "exit":
            continue

        challenger = players[int(usr_input)] 

        ai_turn = False
        if random.uniform(0, 1) >= 0.5:
            ai_turn = True

        board = np.array([np.array([0.0 for i in range(rows)]) for j in range(columns)]).reshape(1, columns, rows, 1)
        visualize_board(board[0], columns, rows)

        game_over = False

        # Play a game against the loaded model
        while(not game_over):
            move = 0

            if ai_turn:
                print("AI Turn:")
        
                options = challenger.predict(board)[0]
                best_columns = []
                best_chance = -float('inf')
                for i in range(len(options)):
                    if options[i] > best_chance:
                        best_columns = [i]
                        best_chance = options[i]
                    elif options[i] >= best_chance:
                        best_columns.append(i)

                move = np.random.choice(best_columns)
                print(move)
            else:
                move = int(input("Your Turn:\nEnter your move (leftmost column is 0)\n"))

            valid_move_check, board = make_move(board, move, not ai_turn)

            visualize_board(board[0], columns, rows)

            game_over, win_num = check_game_over(board[0])

            ai_turn = not ai_turn

        if win_num == 1:
            print("Better luck next time...")
        elif win_num == -1:
            print("Congrats!")
        else:
            print("A ... tie?")

# Calculate the model with the highest winrate against all the other models (very slow)
else:
    players = load_models(args[1])
    top_x = int(args[2])

    stats = []

    for i in range(len(players)):
        print("Evaluating plyaer", i)

        wr = 0.0

        for j in range(len(players)):
            if not players[i] is players[j]:
                p1 = players[i]
                p2 = players[j]

                if np.random.uniform(0.0, 1.0) >= 0.5:
                    p1 = players[j]
                    p2 = players[i]

                winner = play(p1, p2, columns, rows)

                if winner is players[i]:
                    wr += 1

        stats.append(((wr / (len(players) - 1)), i))

    top = heapq.nlargest(top_x, stats)

    for i in range(len(top)):
        print(str(i + 1) + ": Net", top[i])

