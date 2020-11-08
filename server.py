from flask import Flask, json
from flask_cors import CORS, cross_origin
import random
import os

from riotwatcher import LolWatcher, ApiError
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import datetime
import pandas as pd

key = 'RGAPI-4bb6a77f-4f91-4629-a761-b70c65ea0605'
region = 'na1'
watcher = LolWatcher(key)

# load the model from disk
model_name = 'model.sav'
loaded_model = pickle.load(open(model_name, 'rb'))

tiltedness = [
            {"id": 1, "name": "Tilt Level One", "tiltedness": 1},
            {"id": 2, "name": "Tilt Level Two", "tiltedness": 2},
            {"id": 3, "name": "Tilt Level Three", "tiltedness": 3},
            {"id": 4, "name": "Tilt Level Four", "tiltedness": 4},
            {"id": 5, "name": "Tilt Level Five", "tiltedness": 5},
            ]

tilt = ["1", "2", "3", "4", "5"]

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'

@api.route('/gettilt/<summoner>', methods=['GET'])
def get_tiltedness(summoner):
  print(summoner)
  prediction = predict(summoner)
  print ('prediction = ', prediction)
  
  return str(prediction[0])

@api.route('/alltilts', methods=['GET'])
def get_alltilts():
  return json.dumps(tiltedness)

@api.route('/', methods=['GET'])
def get_home():
  return "hello :)"


def is_today(time):
    time /= 1000
    evaluate = datetime.datetime.utcfromtimestamp(time).strftime('%Y-%m-%d')
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    return evaluate == today

def predict(summoner_name):
  player_data = watcher.summoner.by_name(region, summoner_name)
  user_data = get_data(player_data['id'], player_data['accountId'], watcher)
  user_df = pd.DataFrame(user_data)
  processed_df = postprocess(user_df)
  pred_x = np.array(processed_df)
  
  prediction = loaded_model.predict(pred_x)
  print('prediction = ', prediction)
  if prediction is None:
    return "Your tilt-score could not be calculated because you haven't played any games yet today!"
  return prediction

def fix_bools(val_list):
    refactored = []
    for val in val_list:
        if val == 'TRUE':
            refactored.append(1)
        elif val == 'FALSE':
            refactored.append(0)
        elif val == True:
            refactored.append(1)
        elif val == False:
            refactored.append(0)
        else:
            refactored.append(-1)
        
    return refactored

def postprocess(df):
    print ("post processing data")
    # fix labels #
    hotstreak = df['hotstreak'].tolist()
    df['hotstreak'] = fix_bools(hotstreak)

    # fix win_g1 # 
    win_g1 = df['win_g1'].tolist()
    df['win_g1'] = fix_bools(win_g1)
    
    win_g2 = df['win_g2'].tolist()
    df['win_g2'] = fix_bools(win_g2)
    
    win_g3 = df['win_g3'].tolist()
    df['win_g3'] = fix_bools(win_g3)
    
    win_g4 = df['win_g4'].tolist()
    df['win_g4'] = fix_bools(win_g4)
    
    win_g5 = df['win_g5'].tolist()
    df['win_g5'] = fix_bools(win_g5)
        
    return df

def get_data(pid, accountid, watcher):
  # (pid, accountid, watcher):
  '''
  Given player idea, generate a single row of data
  '''
  print ("Pulling data for user")
  region = 'na1'
  mapping_list = [] # stores all mappings
  

  try:
    player_info = watcher.league.by_summoner('na1', pid)
    solo_queue_rank_info = [x for x in player_info if x['queueType'] == 'RANKED_SOLO_5x5'][0]  # query 1

    # compute overal character features #
    win_loss_ratio = float(solo_queue_rank_info['wins']) / solo_queue_rank_info['losses']
    hotstreak = solo_queue_rank_info['hotStreak']

    # compute match features #
    matches = watcher.match.matchlist_by_account(region, accountid)
    
    mapping = {}
    
    # overall features #
    mapping['win_loss_ratio'] = win_loss_ratio
    mapping['hotstreak'] = hotstreak

    for idx, game in enumerate(matches['matches']):
        idx+=1
        if idx < 6:
            if is_today(game['timestamp']):
                # same day, record this data
                game_id = game['gameId']
                match_detail = watcher.match.by_id(region, game_id)
                participant_id = [x for x in match_detail['participantIdentities'] if x['player']['accountId'] == accountid][0]['participantId']

                match_features = [x for x in match_detail['participants'] if x['participantId'] == participant_id][0]
                win = match_features['stats']['win']
                if match_features['stats']['deaths'] == 0:
                    kda = float(match_features['stats']['kills'] + match_features['stats']['assists']) / 1
                else:
                    kda = float(match_features['stats']['kills'] + match_features['stats']['assists']) / match_features['stats']['deaths']
                largest_killing_spree = match_features['stats']['largestKillingSpree']
                longest_time_alive = match_features['stats']['longestTimeSpentLiving']
                try:
                    xp_per_min_delta_1 = match_features['timeline']['xpPerMinDeltas']['0-10']
                except:
                    xp_per_min_delta_1 = -1
                try:
                    xp_per_min_delta_2 = match_features['timeline']['xpPerMinDeltas']['10-20']
                except:
                    xp_per_min_delta_2 = -1              
                try:
                    gold_per_min_delta_1 = match_features['timeline']['goldPerMinDeltas']['0-10']
                except:
                    gold_per_min_delta_1 = -1
                try:
                    gold_per_min_delta_2 = match_features['timeline']['goldPerMinDeltas']['10-20']
                except:
                    gold_per_min_delta_2 = -1
                try:
                    cs_per_min_delta_1 = match_features['timeline']['csDiffPerMinDeltas']['0-10']
                except:
                    cs_per_min_delta_1 = -1
                try: 
                    cs_per_min_delta_2 = match_features['timeline']['csDiffPerMinDeltas']['10-20']
                except:
                    cs_per_min_delta_2 = -1
                            
                # other features that could be used for data analysis later #
                gold_earned = match_features['stats']['goldEarned']
                try:
                    wards_placed = match_features['stats']['wardsPlaced']
                except:
                    wards_placed = -1
                lane = match_features['timeline']['lane']

                # add data
                mapping[f'win_g{idx}'] = win
                mapping[f'kda_g{idx}'] = kda
                mapping[f'largest_killing_spree_g{idx}'] = largest_killing_spree
                mapping[f'longest_time_alive_g{idx}'] = longest_time_alive
                mapping[f'xp_per_min_delta_1_g{idx}'] = xp_per_min_delta_1
                mapping[f'xp_per_min_delta_2_g{idx}'] = xp_per_min_delta_2
                mapping[f'gold_per_min_delta_1_g{idx}'] = gold_per_min_delta_1
                mapping[f'gold_per_min_delta_2_g{idx}'] = gold_per_min_delta_2
                mapping[f'cs_per_min_delta_1_g{idx}'] = cs_per_min_delta_1
                mapping[f'cs_per_min_delta_2_g{idx}'] = cs_per_min_delta_2
            else:
                # no data past g-2, replace with -1
                mapping[f'win_g{idx}'] = -1
                mapping[f'kda_g{idx}'] = -1
                mapping[f'largest_killing_spree_g{idx}'] = -1
                mapping[f'longest_time_alive_g{idx}'] = -1
                mapping[f'xp_per_min_delta_1_g{idx}'] = -1
                mapping[f'xp_per_min_delta_2_g{idx}'] = -1
                mapping[f'gold_per_min_delta_1_g{idx}'] = -1
                mapping[f'gold_per_min_delta_2_g{idx}'] = -1
                mapping[f'cs_per_min_delta_1_g{idx}'] = -1
                mapping[f'cs_per_min_delta_2_g{idx}'] = -1
        else:
            # we only care about past 5 games. If we reach here, reset
            mapping_list.append(mapping)
            print ("Finished pulling data")
            return mapping_list

    return mapping_list
          
  except Exception as e: 
      print(e)
      return None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # api.run(host='0.0.0.0', port=port)
    api.run(debug=True, port=port)