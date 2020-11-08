from flask import Flask, json, send_from_directory
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
            {"id": 1, "name": "Tilt Level One", "tiltedness": 1, 'messages':[
              "Tilt? What's tilt. Keep climbing and show the world your true inner Challenger <3. Check your posture and make sure you aren't slouching to give yourself that little extra boost! Try doing 10 victory push ups after each game you stomp, champ!"
            ]},
            {"id": 2, "name": "Tilt Level Two", "tiltedness": 2, 'messages':[
              "Alright champ, looks like you are slaying it. Odds are, you aren't tilted at all - in fact you are probably feeling great! Remember to drink at leastt 8 cups of water a day to make sure your organs are healthy so you can keep stomping!"
            ]},
            {"id": 3, "name": "Tilt Level Three", "tiltedness": 3, 'messages':[
              "Let's take a breather. There's no need to queue up again right away. Try burning away some of that frustration by doing 30 minutes of exercise - it will help clear your mind and help you win your next few games!"
            ]},
            {"id": 4, "name": "Tilt Level Four", "tiltedness": 4, 'messages':[
              "Looks like you aren't having a great time. We predict that you are PRETTY tilted. Let's take a step away and take out some of that frustration doing some pushups. "
            ]},
            {"id": 5, "name": "Tilt Level Five", "tiltedness": 5, 'messages':[
              "STOP. You are mega tilted. We predict that you are not about to have a fun time if you queue up again. Consider taking some deep breaths and walking away from the game. Have you been drinking water? Consider doing some exercise to burn off some steam. Today might not be your day, but you'll get 'em next time!"
            ]},
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
  
  tilt_object = None
  if prediction > 0.8:
    tilt_object = tiltedness[4]
  elif prediction > 0.6:
    tilt_object = tiltedness[3]
  elif prediction > 0.4:
    tilt_object = tiltedness[2]
  elif prediction > 0.2:
    tilt_object = tiltedness[1]
  else:
    tilt_object = tiltedness[0]

  tilt_object['score'] = prediction
  
  return tilt_object

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
  
  prediction = loaded_model.predict_proba(pred_x)
  print('prediction = ', prediction)
  if prediction is None:
    return "Your tilt-score could not be calculated because you haven't played any games yet today!"
  return prediction[0][1]

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


# --- Graphs Outputs
from plotnine import ggplot, aes, geom_point, geom_line, geom_smooth, scale_x_reverse, geom_text, facet_wrap, xlab, ylab, ylim, ggtitle
def same_day(time_1, time_2):
    time_1 /= 1000
    time_2 /= 1000
    
    evaluate_1 = datetime.datetime.utcfromtimestamp(time_1).strftime('%Y-%m-%d')
    evaluate_2 = datetime.datetime.utcfromtimestamp(time_2).strftime('%Y-%m-%d')
    
    return evaluate_1 == evaluate_2

def graph_get_data(pid, accountid):
    '''
    Given player idea, generate a single row of data
    '''
    region = 'na1'
    mapping_list = [] # stores all mappings
    
    try:
        solo_queue_rank_info = [x for x in watcher.league.by_summoner(region, pid) if x['queueType'] == 'RANKED_SOLO_5x5'][0]  # query 1

        # compute match features #
        matches = watcher.match.matchlist_by_account(region, accountid)
        eval_day = None
        idx = 0

        for game in matches['matches']:
            print ('idx = ', idx)
            mapping = {}
            if idx == 0:
                # gen label #
                eval_day = game['timestamp']
                game_id = game['gameId']
                match_detail = watcher.match.by_id(region, game_id)
                game_length_sec=match_detail['gameDuration']
                participant_id = [x for x in match_detail['participantIdentities'] if x['player']['accountId'] == accountid][0]['participantId']
                match_features = [x for x in match_detail['participants'] if x['participantId'] == participant_id][0]
                label = match_features['stats']['win']

                idx+=1
            elif idx < 6:
                if same_day(game['timestamp'], eval_day):
                    # same day, record this data
                    game_id = game['gameId']
                    match_detail = watcher.match.by_id(region, game_id)
                    participant_id = [x for x in match_detail['participantIdentities'] if x['player']['accountId'] == accountid][0]['participantId']
                    match_features = [x for x in match_detail['participants'] if x['participantId'] == participant_id][0]

                    win = match_features['stats']['win']

                    mapping['win'] = match_features['stats']['win']
                    mapping['kills'] = match_features['stats']['kills']
                    mapping['deaths'] = match_features['stats']['deaths']
                    mapping['assists'] = match_features['stats']['assists']
                    mapping['kda'] = match_features['stats']['kills'] + match_features['stats']['assists'] / match_features['stats']['deaths']
                    mapping['totalDamageDealtToChampions'] = match_features['stats']['totalDamageDealtToChampions']
                    mapping['magicDamageDealtToChampions'] = match_features['stats']['magicDamageDealtToChampions']
                    mapping['physicalDamageDealtToChampions'] = match_features['stats']['physicalDamageDealtToChampions']

                    mapping['TotalDamageDealtToChampionsPerMinute'] = match_features['stats']['totalDamageDealtToChampions']/game_length_sec*60
                    mapping['MagicDamageDealtToChampionsPerMinute'] = match_features['stats']['magicDamageDealtToChampions']/game_length_sec*60
                    mapping['PhysicalDamageDealtToChampionsPerMinute'] = match_features['stats']['physicalDamageDealtToChampions']/game_length_sec*60

                    mapping['visionScore'] = match_features['stats']['visionScore']

                    mapping['goldEarned'] = match_features['stats']['goldEarned']
                    mapping['goldEarnedPerMinute'] = match_features['stats']['goldEarned']/game_length_sec*60

                    mapping['totalMinionsKilled'] = match_features['stats']['totalMinionsKilled']
                    mapping['totalMinionsKilledPerMinute'] = match_features['stats']['totalMinionsKilled']/game_length_sec*60

                    mapping['role'] = match_features['timeline']['role']
                    mapping['lane'] = match_features['timeline']['lane']

                    mapping_list.append(mapping)

                    idx+=1
                elif idx == 1:
                    # if we are looking at the second game, and it is not in the same day, we have no good data. Do not record anything
                    # reset and start over
                    eval_day = game['timestamp']
                    idx = 0
                else:
                    # no data past g-2, replace with -1
                    if idx < 5:
                        mapping['win'] = -1
                        mapping['kills'] = -1
                        mapping['deaths'] = -1
                        mapping['assists'] = -1
                        mapping['kda'] = -1
                        mapping['totalDamageDealtToChampions'] = -1
                        mapping['magicDamageDealtToChampions'] = -1
                        mapping['physicalDamageDealtToChampions'] = -1

                        mapping['TotalDamageDealtToChampionsPerMinute'] = -1
                        mapping['MagicDamageDealtToChampionsPerMinute'] = -1
                        mapping['PhysicalDamageDealtToChampionsPerMinute'] = -1

                        mapping['visionScore'] = -1

                        mapping['goldEarned'] = -1
                        mapping['goldEarnedPerMinute'] = -1

                        mapping['totalMinionsKilled'] = -1
                        mapping['totalMinionsKilledPerMinute'] = -1

                        mapping['role'] = -1
                        mapping['lane'] = -1
                    idx+=1
                    mapping_list.append(mapping)
            else:
                # we only care about past 5 games. If we reach here, reset
                break

        return mapping_list

    except Exception as e: 
        print(e)
        print("no solo queue info")
        return None

def graph_process_player():

    return

@api.route('/dummy/<summoner>', methods=['GET'])
def graph_get_history(summoner):
    player = watcher.summoner.by_name('na1', summoner)
    mapping_list = graph_get_data(player['id'], player['accountId'])
    print(mapping_list)
    master_list = []
    master_list.extend(mapping_list)
    df=pd.DataFrame(master_list)
    # Fix Support & ADC Labels
    df.loc[df.role == 'DUO_SUPPORT', 'lane'] = "SUPPORT"
    df.loc[df.role == 'DUO_CARRY', 'lane'] = "ADC"
    # Create Column Games_Ago for X Axis Ordering
    df['games_ago'] = df.reset_index().index+1
    kda = graph_kda(df)
    kda.save(filename = f'./static/images/{summoner}_kda.jpg', height=3, width=5, units = 'in', dpi=1000)
    tddtc = graph_tddtc(df)
    tddtc.save(filename = f'./static/images/{summoner}_tddtc.jpg', height=3, width=5, units = 'in', dpi=1000)
    mddtc = graph_mddtc(df)
    mddtc.save(filename = f'./static/images/{summoner}_mddtc.jpg', height=3, width=5, units = 'in', dpi=1000)
    pddtc = graph_pddtc(df)
    pddtc.save(filename = f'./static/images/{summoner}_pddtc.jpg', height=3, width=5, units = 'in', dpi=1000)
    gepm = graph_gepm(df)
    gepm.save(filename = f'./static/images/{summoner}_gepm.jpg', height=3, width=5, units = 'in', dpi=1000)
    tmk = graph_tmk(df)
    tmk.save(filename = f'./static/images/{summoner}_tmk.jpg', height=3, width=5, units = 'in', dpi=1000)
    return 'Files Saved!'

def graph_kda(df):
    kda = ggplot(df) + aes(x='games_ago', y='kda') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + ylim(0,30) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('KDA') + ggtitle("KDA Over the Last 5 Games")
    return kda

def graph_tddtc(df):
    tddtc = ggplot(df) + aes(x='games_ago', y='TotalDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Total Damage Dealt To Champions \n Per Minute') + ggtitle("Total Damage Dealt To Champions Per Minute \n Over the Last 5 Games")
    return tddtc

def graph_mddtc(df):
    mddtc = ggplot(df) + aes(x='games_ago', y='MagicDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Magic Damage Dealt To Champions \n Per Minute') + ggtitle("Magic Damage Dealt To Champions Per Minute \n Over the Last 5 Games")   
    return mddtc

def graph_pddtc(df):
    pddtc = ggplot(df) + aes(x='games_ago', y='PhysicalDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Physical Damage Dealt To Champions \n Per Minute') + ggtitle("Physical Damage Dealt To Champions Per Minute \n Over the Last 5 Games")
    return pddtc

def graph_gepm(df):
    gepm = ggplot(df) + aes(x='games_ago', y='goldEarnedPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Gold Earned Per Minute') + ggtitle("Gold Earned Per Minute \n Over the Last 5 Games")
    return gepm

def graph_tmk(df):
    tmk = ggplot(df) + aes(x='games_ago', y='totalMinionsKilledPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Total Minions Killed \n Per Minute') + ggtitle("Total Minions Killed Per Minute \n Over the Last 5 Games")
    return tmk

@api.route('/img/<summoner>')
def send_img(summoner):
    return send_from_directory('./static/images', f'{summoner}.jpg') ## need to fix this

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # api.run(host='0.0.0.0', port=port)
    api.run(debug=True, port=port)