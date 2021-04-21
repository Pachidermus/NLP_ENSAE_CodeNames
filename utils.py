from game import *
import itertools
import numpy as np
import pandas as pd

def bot_benchmark_run(teams=None, seeds = [1, 2, 3, 4, 5]):
    
    if teams is None:
        print("Please define teams")
        return None
    
    Game.clear_results()
    
    red_team = {'red_team': True}
    blue_team = {'red_team': False}

    for team in teams:
        
        cm_A = team[0]
        g_A = team[1]
        name_A = team[2]
        
        cm_B = team[0]
        g_B = team[1]
        name_B = team[2]
        
        for seed in seeds:
            
            game_name = f"red_{name_A}_VS_blue_{name_B}_seed_{seed}"
            print(game_name)
            Game2(cm_A[0], g_A[0], cm_B[0], g_B[0], 
                  seed = seed, 
                  do_print = False,  
                  cmr_kwargs = {**cm_A[1], **red_team},
                  gr_kwargs = g_A[1],
                  cmb_kwargs = {**cm_B[1], **blue_team},
                  gb_kwargs = g_B[1],
                  game_name = game_name).run()
            
    if len(teams) > 1:
        
        for combination in itertools.combinations(teams, 2):
            team_A = combination[0]
            team_B = combination[1]

            cm_A = team_A[0]
            g_A = team_A[1]
            name_A = team_A[2]

            cm_B = team_B[0]
            g_B = team_B[1]
            name_B = team_B[2]

            for seed in seeds:

                game_name = f"red_{name_A}_VS_blue_{name_B}_seed_{seed}"
                print(game_name)
                Game2(cm_A[0], g_A[0], cm_B[0], g_B[0], 
                      seed = seed, 
                      do_print = False,  
                      cmr_kwargs = {**cm_A[1], **red_team},
                      gr_kwargs = g_A[1],
                      cmb_kwargs = {**cm_B[1], **blue_team},
                      gb_kwargs = g_B[1],
                      game_name = game_name).run()

                game_name = f"red_{name_B}_VS_blue_{name_A}_seed_{seed}"
                print(game_name)
                Game2(cm_B[0], g_B[0], cm_A[0], g_A[0], 
                      seed = seed, 
                      do_print = False,  
                      cmr_kwargs = {**cm_B[1], **red_team},
                      gr_kwargs = g_B[1],
                      cmb_kwargs = {**cm_A[1], **blue_team},
                      gb_kwargs = g_A[1],
                      game_name = game_name).run()

def bot_benchmark_analysis_overview(verbose = True, path = "results/"):

    ##RESULTS READING
    game_names = []
    
    cmr_names = []
    gr_names= []
    cmb_names = []
    gb_names= []
    
    
    winners = []
    times = []
    turns = []
    times_fg = []
    turns_fg = []

    red_wins = 0
    red_scores = []
    red_scores_fg = []
    blue_wins = 0
    blue_scores = []
    blue_scores_fg = []
    civilian_reveals = []
    assassins_called = 0
    
    with open(path + "bot_results_new_style.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            
            game_names.append(game_json["game_name"])
            
            cmr_names.append(game_json["codemaster_red"])
            gr_names.append(game_json["guesser_red"])
            cmb_names.append(game_json["codemaster_blue"])
            gb_names.append(game_json["guesser_blue"])
            
            winners.append(game_json["winner"])
            turns.append(game_json["total_turns"])
            times.append(game_json["time_s"])

            red_scores.append(game_json["R"])
            blue_scores.append(game_json["B"])
            
            civilian_reveals.append(game_json["C"])

            assassins_called += game_json["A"]

            if game_json["winner"] == 'Red':
                red_wins += 1

            elif game_json["winner"] == 'Blue':
                blue_wins += 1

            if game_json["A"] == 0:
                turns_fg.append(game_json["total_turns"])
                times_fg.append(game_json["time_s"])

                red_scores_fg.append(game_json["R"])
                blue_scores_fg.append(game_json["B"])

    ##PRINT OF OVERALL RESULTS
    
    nb_games = red_wins + blue_wins
    if verbose:
    
        print("#-#-#Overall Results#-#-#")
        print("\n")
        print("-#-On all games-#-")
        print("\n")
        print(f"Ratio of Red wins: {round(100*red_wins/nb_games,2)}%")
        print(f"Ratio of Blue wins: {round(100*blue_wins/nb_games,2)}%")
        print(f"Ratio of Assassins called: {round(100*assassins_called/nb_games,2)}%")
        print(f"Average red score: {round(np.mean(red_scores),2)} words")
        print(f"Average blue score: {round(np.mean(blue_scores),2)} words")
        print(f"Average civilians: {round(np.mean(civilian_reveals),2)} words")
        print(f"Average number of turns: {round(np.mean(turns),2)} turns")
        print(f"Average game time: {round(np.mean(times),2)} seconds")
        print("\n")
        print("-#-On games where the assassin has been avoided-#-")
        print("\n")
        print(f"Average red score: {round(np.mean(red_scores_fg),2)} words")
        print(f"Average blue score: {round(np.mean(blue_scores_fg),2)} words")

        print(f"Average number of turns: {round(np.mean(turns_fg),2)} turns")
        print(f"Average game time: {round(np.mean(times_fg),2)} seconds")
        print("\n")
    
    all_games_dict = {'game_name': game_names, 
            'cmr_name': cmr_names, 'gr_name': gr_names,
            'cmb_name': cmb_names, 'gb_name': gb_names,
            'winner': winners,
            'red_score': red_scores, 'blue_score': blue_scores, 
            'turn': turns, 'time': times}
    
    all_games = pd.DataFrame(all_games_dict)
    
    return all_games

def bot_benchmark_analysis_marginal(all_games):
    
    results = all_games
    
    g_names = np.unique(results["gr_name"]).tolist()
    cm_names = np.unique(results["cmr_name"]).tolist()

    names = []
    types = []
    nb_win_as_red = []
    nb_win_as_blue = []
    avg_score_as_red = []
    avg_score_as_blue= []
    nb_win = []
    win_ratio = []
    nb_assassins = []
    assassins_ratio = []
    avg_turns = []
    avg_turns_no_assassin = []
    avg_times = []
    avg_times_no_assassin = []
    
    ##GUESSERS ANALYSIS
    for g_name in g_names:

        all_appearances = results[(results['gr_name'] == g_name) | (results['gb_name'] == g_name)]

        total_games = results[results['gr_name'] == g_name].shape[0] + results[results['gb_name'] == g_name].shape[0]
        names.append(g_name)
        types.append("Guesser")
        nb_win_as_red.append(np.sum(results[results['gr_name'] == g_name]["winner"] == "Red"))
        avg_score_as_red.append(np.mean(results[results['gr_name'] == g_name]["red_score"]))
        nb_win_as_blue.append(np.sum(results[results['gb_name'] == g_name]["winner"] == "Blue"))
        avg_score_as_blue.append(np.mean(results[results['gb_name'] == g_name]["blue_score"]))
        nb_win.append(nb_win_as_red[-1] + nb_win_as_blue[-1])
        win_ratio.append((nb_win[-1]) / total_games)

        assassins_called_as_red = results[(results['gr_name'] == g_name) & (results["blue_score"]<7) & (results["winner"]== "Blue")].shape[0]
        assassins_called_as_blue = results[(results['gb_name'] == g_name) & (results["red_score"]<8) & (results["winner"]== "Red")].shape[0]
        nb_assassins.append(assassins_called_as_red + assassins_called_as_blue)
        assassins_ratio.append((nb_assassins[-1]) / total_games)
        
        avg_turns.append(np.mean(all_appearances['turn']))
        avg_turns_no_assassin.append(np.mean(all_appearances[np.logical_or(all_appearances['red_score'] == 8 , all_appearances['blue_score'] == 7)]['turn']))
        avg_times.append(np.mean(all_appearances['time']))
        avg_times_no_assassin.append(np.mean(all_appearances[np.logical_or(all_appearances['red_score'] == 8 , all_appearances['blue_score'] == 7)]['time']))
    
    ##CODEMASTERS ANALYSIS
    for cm_name in cm_names:

        all_appearances = results[(results['cmr_name'] == cm_name) | (results['cmb_name'] == cm_name)]

        total_games = results[results['cmr_name'] == cm_name].shape[0] + results[results['cmb_name'] == cm_name].shape[0]
        names.append(cm_name)
        types.append("CodeMaster")
        nb_win_as_red.append(np.sum(results[results['cmr_name'] == cm_name]["winner"] == "Red"))
        avg_score_as_red.append(np.mean(results[results['cmr_name'] == cm_name]["red_score"]))
        nb_win_as_blue.append(np.sum(results[results['cmb_name'] == cm_name]["winner"] == "Blue"))
        avg_score_as_blue.append(np.mean(results[results['cmb_name'] == cm_name]["blue_score"]))
        nb_win.append(nb_win_as_red[-1] + nb_win_as_blue[-1])
        win_ratio.append((nb_win[-1]) / total_games)
        
        assassins_called_as_red = results[(results['cmr_name'] == cm_name) & (results["blue_score"]<7) & (results["winner"] == "Blue")].shape[0]
        assassins_called_as_blue = results[(results['cmb_name'] == cm_name) & (results["red_score"]<8) & (results["winner"] == "Red")].shape[0]
        nb_assassins.append(assassins_called_as_red + assassins_called_as_blue)
        assassins_ratio.append((nb_assassins[-1]) / total_games)
        
        avg_turns.append(np.mean(all_appearances['turn']))
        avg_turns_no_assassin.append(np.mean(all_appearances[np.logical_or(all_appearances['red_score'] == 8 , all_appearances['blue_score'] == 7)]['turn']))
        avg_times.append(np.mean(all_appearances['time']))
        avg_times_no_assassin.append(np.mean(all_appearances[np.logical_or(all_appearances['red_score'] == 8 , all_appearances['blue_score'] == 7)]['time']))


    marginal_analysis_dict = {'name': names, 'type': types,
                       'nb_win_as_red': nb_win_as_red, 'nb_win_as_blue': nb_win_as_blue,
                       'avg_score_as_red': avg_score_as_red, 'avg_score_as_blue': avg_score_as_blue,
                       'nb_win': nb_win, 'win_ratio': win_ratio,
                       'nb_assassin': nb_assassins, 'assassin_ratio': assassins_ratio,
                       'avg_turn': avg_turns, 'avg_time': avg_times,
                       'avg_turn_no_assassin': avg_turns_no_assassin, 'avg_time_no_assassin': avg_times_no_assassin}
    
    marginal_analysis = pd.DataFrame(marginal_analysis_dict)
    
    return marginal_analysis

def bot_benchmark_analysis_marginal_team(all_games):
    
    results = all_games
    
    t_names = np.unique(results["game_name"].apply(lambda x: x.split('_')[1]))

    names = []
    types = []
    nb_win_as_red = []
    nb_win_as_blue = []
    avg_score_as_red = []
    avg_score_as_blue= []
    nb_win = []
    win_ratio = []
    nb_assassins = []
    assassins_ratio = []
    avg_turns = []
    avg_turns_no_assassin = []
    avg_times = []
    avg_times_no_assassin = []
    
    ##TEAMS ANALYSIS
    for t_name in t_names:

        subresults = results[results['game_name'].apply(lambda x: t_name in x)]

        
        t_as_red_subset = subresults[subresults['game_name'].apply(lambda x: 'red_' + t_name in x)]
        t_as_blue_subset = subresults[subresults['game_name'].apply(lambda x: 'blue_' + t_name in x)]

        total_games = t_as_red_subset.shape[0] + t_as_blue_subset.shape[0]
        names.append(t_name)
        types.append("Team")
        nb_win_as_red.append(np.sum(t_as_red_subset["winner"] == "Red"))
        avg_score_as_red.append(np.mean(t_as_red_subset["red_score"]))
        nb_win_as_blue.append(np.sum(t_as_blue_subset["winner"] == "Blue"))
        avg_score_as_blue.append(np.mean(t_as_blue_subset["blue_score"]))
        nb_win.append(nb_win_as_red[-1] + nb_win_as_blue[-1])
        win_ratio.append((nb_win[-1]) / total_games)

        assassins_called_as_red = t_as_red_subset[(t_as_red_subset["blue_score"]<7) & (t_as_red_subset["winner"]== "Blue")].shape[0]
        assassins_called_as_blue = t_as_blue_subset[(t_as_blue_subset["red_score"]<8) & (t_as_blue_subset["winner"]== "Red")].shape[0]
        nb_assassins.append(assassins_called_as_red + assassins_called_as_blue)
        assassins_ratio.append((nb_assassins[-1]) / total_games)
        
        avg_turns.append(np.mean(subresults['turn']))
        avg_turns_no_assassin.append(np.mean(subresults[np.logical_or(subresults['red_score'] == 8 , subresults['blue_score'] == 7)]['turn']))
        avg_times.append(np.mean(subresults['time']))
        avg_times_no_assassin.append(np.mean(subresults[np.logical_or(subresults['red_score'] == 8 , subresults['blue_score'] == 7)]['time']))

    marginal_analysis_dict = {'name': names, 'type': types,
                       'nb_win_as_red': nb_win_as_red, 'nb_win_as_blue': nb_win_as_blue,
                       'avg_score_as_red': avg_score_as_red, 'avg_score_as_blue': avg_score_as_blue,
                       'nb_win': nb_win, 'win_ratio': win_ratio,
                       'nb_assassin': nb_assassins, 'assassin_ratio': assassins_ratio,
                       'avg_turn': avg_turns, 'avg_time': avg_times,
                       'avg_turn_no_assassin': avg_turns_no_assassin, 'avg_time_no_assassin': avg_times_no_assassin}
    
    marginal_analysis = pd.DataFrame(marginal_analysis_dict)
    
    return marginal_analysis

def bot_benchmark_analysis_competitive(all_games, teams_names):
    
    results = all_games
    
    team_name_a = []
    team_name_b = []
    win_ratio_a = []
    win_ratio_b = []
    nb_assassins_called = []
    nb_turns = []
    nb_turns_no_assassin = []
    avg_time = []
    avg_time_no_assassin = []
    
    for team_name_comb in itertools.combinations(teams_names, 2):
        
        tn_a = team_name_comb[0]
        tn_b = team_name_comb[1]
        team_name_a.append(tn_a)
        team_name_b.append(tn_b)
        
        subresults = results[results['game_name'].apply(lambda x: tn_a in x and tn_b in x)]
        nb_seeds = subresults.shape[0]
        
        a_as_red_subset = subresults[subresults['game_name'].apply(lambda x: 'red_' + tn_a in x)]
        a_as_blue_subset = subresults[subresults['game_name'].apply(lambda x: 'blue_' + tn_a in x)]
        b_as_red_subset = subresults[subresults['game_name'].apply(lambda x: 'red_' + tn_b in x)]
        b_as_blue_subset = subresults[subresults['game_name'].apply(lambda x: 'blue_' + tn_b in x)]
        
        a_win_as_red = np.sum(a_as_red_subset['winner']=='Red')
        a_win_as_blue = np.sum(a_as_blue_subset['winner']=='Blue')
        a_win = a_win_as_red + a_win_as_blue
        
        win_ratio_a.append(a_win / nb_seeds)
        win_ratio_b.append(1 - win_ratio_a[-1])
        nb_assassins_called.append(np.sum(np.logical_and(subresults['red_score'] < 8 , subresults['blue_score'] < 7)))
        nb_turns.append(np.mean(subresults['turn']))
        avg_time.append(np.mean(subresults['time']))
        
        nb_turns_no_assassin.append(np.mean(subresults[np.logical_or(subresults['red_score'] == 8 , subresults['blue_score'] == 7)]['turn']))
        avg_time_no_assassin.append(np.mean(subresults[np.logical_or(subresults['red_score'] == 8 , subresults['blue_score'] == 7)]['time']))
        
        
    competitive_analysis_dict = {'team_name_a': team_name_a, 'team_name_b': team_name_b,
                                'win_ratio_a': win_ratio_a, 'win_ratio_b': win_ratio_b,
                                'nb_assassin': nb_assassins_called,
                                'avg_turn': nb_turns, 'avg_time' : avg_time,
                                'avg_turn_no_assassin': nb_turns_no_assassin, 'avg_time_no_assassin' : avg_time_no_assassin}
    
    competitive_analysis = pd.DataFrame(competitive_analysis_dict)

    return competitive_analysis

def bot_benchmark_analysis(teams_names, verbose = True, path = "results/"):
    
    #OVERALL ANALYSIS
    all_games = bot_benchmark_analysis_overview(verbose = verbose, path = path)
    
    #MARGINAL ANALYSIS
    marginal_analysis1 = bot_benchmark_analysis_marginal(all_games)
    marginal_analysis2 = bot_benchmark_analysis_marginal_team(all_games)
    
    #COMPETITIVE ANALYSIS
    competitive_analysis = bot_benchmark_analysis_competitive(all_games, teams_names)
    
    return all_games, marginal_analysis1, marginal_analysis2, competitive_analysis