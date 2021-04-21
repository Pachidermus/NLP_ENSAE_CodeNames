import time
import json
from game import Game2
from players.guesser import *
from players.codemaster import *

class all_humans_Simple_Example:
    """Example of how to share vectors, pass kwargs, and call Game directly instead of by terminal"""

    start_time = time.time()
    
    print("\nclearing results folder...\n")
    Game2.clear_results()

    seed = 0

    #
    print("starting original all humans game")
    Game2(HumanCodemaster, HumanGuesser, HumanCodemaster, HumanGuesser, seed=seed, do_print=True,  game_name="all humans").run()

    # display the results
    print(f"\nfor seed {seed} ~")
    with open("results/bot_results_new_style.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            game_name = game_json["game_name"]
            game_time = game_json["time_s"]
            game_score = game_json["total_turns"]

            print(f"time={game_time:.2f}, turns={game_score}, name={game_name}")


if __name__ == "__main__":
    all_humans_Simple_Example()