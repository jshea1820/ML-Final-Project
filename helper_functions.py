import math, random, csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def add_win_column(scoreRow):  
    # Make a column that represents if the home team won, tied or loss
    # home win = 1, home loss = -1, draw = 0
    
    if scoreRow[0] > scoreRow[1]:
        return 1
    elif scoreRow[0] == scoreRow[1]:
        return 0
    else:
        return -1

def create_game(df, home, away, date_val, window):
    # Return a df of averaged stats from previous 5 games that represents a predicted game

    new_game = {'date_value': date_val, 'home_team': home, 'away_team': away, 'home_possession': 0, 'away_possession': 0, 'home_pass_acc': 0, 'away_pass_acc': 0, 'home_sot': 0, 'away_sot': 0, 'home_saves': 0,
                 'away_saves': 0, 'home_fouls': 0, 'away_fouls': 0, 'home_corners': 0, 'away_corners': 0, 'home_crosses': 0, 'away_crosses': 0, 'home_touches': 0, 'away_touches': 0, 'home_tackles': 0,
                  'away_tackles': 0, 'home_ints': 0, 'away_ints': 0, 'home_aerials': 0, 'away_aerials': 0, 'home_clearances': 0, 'away_clearances': 0, 'home_offsides': 0, 'away_offsides': 0, 'home_goal_kicks': 0, 'away_goal_kicks': 0,
                  'home_throwins': 0, 'away_throwins': 0, 'home_longballs': 0, 'away_longballs': 0,}

   
    # Extract and sum up previous 5 games from Home team
    num_found = 0
    for index, data in df.iterrows():
        if int(data['date_value']) >= int(date_val): continue  # Skip until we get to the point in time we are predicting for
        if num_found >= window: break

        if data['home_team'] == home:
            new_game['home_possession'] += data['home_possession']
            new_game['home_pass_acc'] += data['home_pass_acc']
            new_game['home_sot'] += data['home_sot']
            new_game['home_saves'] += data['home_saves']
            new_game['home_fouls'] += data['home_fouls']
            new_game['home_corners'] += data['home_corners']
            new_game['home_crosses'] += data['home_crosses']
            new_game['home_touches'] += data['home_touches']
            new_game['home_tackles'] += data['home_tackles']
            new_game['home_ints'] += data['home_ints']
            new_game['home_aerials'] += data['home_aerials']
            new_game['home_clearances'] += data['home_clearances']
            new_game['home_offsides'] += data['home_offsides']
            new_game['home_goal_kicks'] += data['home_goal_kicks']
            new_game['home_throwins'] += data['home_throwins']
            new_game['home_longballs'] += data['home_longballs']
        elif data['away_team'] == home:
            new_game['home_possession'] += data['away_possession']
            new_game['home_pass_acc'] += data['away_pass_acc']
            new_game['home_sot'] += data['away_sot']
            new_game['home_saves'] += data['away_saves']
            new_game['home_fouls'] += data['away_fouls']
            new_game['home_corners'] += data['away_corners']
            new_game['home_crosses'] += data['away_crosses']
            new_game['home_touches'] += data['away_touches']
            new_game['home_tackles'] += data['away_tackles']
            new_game['home_ints'] += data['away_ints']
            new_game['home_aerials'] += data['away_aerials']
            new_game['home_clearances'] += data['away_clearances']
            new_game['home_offsides'] += data['away_offsides']
            new_game['home_goal_kicks'] += data['away_goal_kicks']
            new_game['home_throwins'] += data['away_throwins']
            new_game['home_longballs'] += data['away_longballs']
        else: continue

        num_found += 1

    # Do the same for the Away Team
    num_found = 0
    for index, data in df.iterrows():
        if int(data['date_value']) >= int(date_val): continue  # Skip until we get to the point in time we are predicting for
        if num_found >= window: break

        if data['home_team'] == away:
            new_game['away_possession'] += data['home_possession']
            new_game['away_pass_acc'] += data['home_pass_acc']
            new_game['away_sot'] += data['home_sot']
            new_game['away_saves'] += data['home_saves']
            new_game['away_fouls'] += data['home_fouls']
            new_game['away_corners'] += data['home_corners']
            new_game['away_crosses'] += data['home_crosses']
            new_game['away_touches'] += data['home_touches']
            new_game['away_tackles'] += data['home_tackles']
            new_game['away_ints'] += data['home_ints']
            new_game['away_aerials'] += data['home_aerials']
            new_game['away_clearances'] += data['home_clearances']
            new_game['away_offsides'] += data['home_offsides']
            new_game['away_goal_kicks'] += data['home_goal_kicks']
            new_game['away_throwins'] += data['home_throwins']
            new_game['away_longballs'] += data['home_longballs']
        elif data['away_team'] == away:
            new_game['away_possession'] += data['away_possession']
            new_game['away_pass_acc'] += data['away_pass_acc']
            new_game['away_sot'] += data['away_sot']
            new_game['away_saves'] += data['away_saves']
            new_game['away_fouls'] += data['away_fouls']
            new_game['away_corners'] += data['away_corners']
            new_game['away_crosses'] += data['away_crosses']
            new_game['away_touches'] += data['away_touches']
            new_game['away_tackles'] += data['away_tackles']
            new_game['away_ints'] += data['away_ints']
            new_game['away_aerials'] += data['away_aerials']
            new_game['away_clearances'] += data['away_clearances']
            new_game['away_offsides'] += data['away_offsides']
            new_game['away_goal_kicks'] += data['away_goal_kicks']
            new_game['away_throwins'] += data['away_throwins']
            new_game['away_longballs'] += data['away_longballs']
        else: continue

        num_found += 1


    # Average out new game according to window size
    not_ints = ["date_value", "home_team", "away_team"]
    for key, val in new_game.items():
        if key in not_ints: continue
        new_game[key] = val // window 
    
    new_df = pd.DataFrame.from_records([new_game])
    return new_df