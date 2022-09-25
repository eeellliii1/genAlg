from time import sleep
import numpy as np
import os
import random
import sys
import heapq

####

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

####

from game import Connect_4

# TODO 
# 1. Individual weight adjustment
# 2. 'Terminator' Minimax competitors
# 3. Variable population size
# 4. Test on boards of dimensions other than training?


# Splice net weights together and generate new generations #TODO NEEDS ADJUST TO BE MODULE AGNOSTIC
def reproduce(parents, g):
    model = g.make_player().net # NEED TO ELIMINATE THIS just edit weights of the existing models

    # Pick out 2 parents (may adjust later to just take from anything in the parent pool)
    pars = []
    if len(parents) == 2:
        pars = [parents[0], parents[1]] # If there are only 2 parents take those
    else:
        pars = np.random.choice(parents, 2) # Take 2 random nets to be parents

    par_nets = [p.net for p in pars]

    # Mutate value for multiplying weights by mutate_val -> 1 + mutate_val
    mutate_val = 0.5

    # Get the parent model layers
    layers = model.layers
    p1_layers = par_nets[0].layers
    p2_layers = par_nets[1].layers

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

    return g.player(model)


# Load all the models in the given file_path to a folder and return list of players who use 
def load_models(file_path, g):
    file_names = os.listdir(file_path)

    players = []
    for file_name in file_names:
        players.append(g.make_player(tf.keras.models.load_model(file_path + "/" + file_name)))

    return players


# Pit random pairs of models against eachother returning only the ones who win
def tournament(players, g):
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

        winners.append(g.play(p1, p2))

        round += 1

    return winners

#########################################

args = sys.argv[1:]
# Command Struct 1: train {new/old} {load_dir} {save_dir} {generations} {(not needed if old) num_players}
#   only setup right now with generation size divisible by 4
# Command Struct 2: evaluate {load_dir}
#   will assign each net in the dir a number between 0 and the total - 1 in order of how they are listed in the dir
# Command Struct 3: best {load_dir} {top_how_many}
#   finds the net with the best win rate against all other nets

g = Connect_4() # NEEEEEEEEDS FIIIIIIIIIIIIIIX ###################################

# Trains nets either new or loaded
if args[0] == "train":
    num_players = 0 # Number of nets per generation
    generations = int(args[4]) 

    

    # Make brand new nets
    if args[1] == "new":
        num_players = int(args[5])
        players = [g.make_player() for i in range(num_players)]
    # Load previously generated nets
    else: 
        players = load_models(args[2], g)
        num_players = len(players)

    # Train the nets for {generations} generations
    for i in range(generations):
        print("Generation:", i + 1)

        # Play through two halvings of the population (needs changing so input does not need to be a multiple of 4)
        winners = tournament(players, g)
        winners_winners = tournament(winners, g)

        players = [reproduce(winners_winners, g) for i in range(num_players)]

        # Every x generations save the models
        x = 60
        if (i + 1) % x == 0:
            for i in range(len(players)):
                players[i].save(args[3] + '/net' + str(i) + '.h5')

    for i in range(len(players)):
        players[i].save(args[3] + '/net' + str(i) + '.h5')

# Lets you play against the model of your choice 
elif args[0] == "evaluate":
    players = load_models(args[1], g)

    usr_input = ""
    while(usr_input != "exit"):

        usr_input = input("Choose a challenger: 0 -> " + str(len(players) - 1))
        if usr_input == "exit":
            continue

        challenger = players[int(usr_input)] 

        victor = g.evaluate(challenger)

        if victor == challenger:
            print("Better luck next time...")
        else:
            print("Congrats!")

        # ai_turn = False
        # if random.uniform(0, 1) >= 0.5:
        #     ai_turn = True

        # board = np.array([np.array([0.0 for i in range(rows)]) for j in range(columns)]).reshape(1, columns, rows, 1)
        # visualize_board(board[0], columns, rows)

        # game_over = False

        # # Play a game against the loaded model
        # while(not game_over):
        #     move = 0

        #     if ai_turn:
        #         print("AI Turn:")
        
        #         options = challenger.predict(board)[0]
        #         best_columns = []
        #         best_chance = -float('inf')
        #         for i in range(len(options)):
        #             if options[i] > best_chance:
        #                 best_columns = [i]
        #                 best_chance = options[i]
        #             elif options[i] >= best_chance:
        #                 best_columns.append(i)

        #         move = np.random.choice(best_columns)
        #         print(move)
        #     else:
        #         move = int(input("Your Turn:\nEnter your move (leftmost column is 0)\n"))

        #     valid_move_check, board = make_move(board, move, not ai_turn)

        #     visualize_board(board[0], columns, rows)

        #     game_over, win_num = check_game_over(board[0])

        #     ai_turn = not ai_turn

        # if win_num == 1:
        #     print("Better luck next time...")
        # elif win_num == -1:
        #     print("Congrats!")
        # else:
        #     print("A ... tie?")

# Calculate the model with the highest winrate against all the other models (very slow)
elif args[0] == "best": 
    players = load_models(args[1], g)
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

                winner = g.play(p1, p2)

                if winner is players[i]:
                    wr += 1

        stats.append(((wr / (len(players) - 1)), i))

    top = heapq.nlargest(top_x, stats)

    for i in range(len(top)):
        print(str(i + 1) + ": Net", top[i])

