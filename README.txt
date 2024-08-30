Purpose:
-Make a framework for generating and evaluating genetic algorithms against simple adversarial games

Reqs:
-

Command Format:
-Command Struct 1: train {new/old} {load_dir} {save_dir} {generations} {(not needed if old) num_players}
    only setup right now with generation size divisible by 4
-Command Struct 2: evaluate {load_dir}
    will assign each net in the dir a number between 0 and the total - 1 in order of how they are listed in the dir
-Command Struct 3: best {load_dir} {top_how_many}
    finds the net with the best win rate against all other nets