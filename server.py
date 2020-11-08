from flask import Flask, json, send_from_directory, after_this_request
from flask_cors import CORS, cross_origin
import random
import os
import glob
from os import environ

from riotwatcher import LolWatcher, ApiError
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import datetime
import pandas as pd

# load the model from disk
model_name = 'model.sav'
loaded_model = pickle.load(open(model_name, 'rb'))

tiltedness = [
            {"id": 1, "name": "Tilt Level 1", "tiltedness": 1, 'messages':[
              "Tilt? What's tilt. Keep climbing and show the world your true inner Challenger <3. Check your posture and make sure you aren't slouching to give yourself that little extra boost! Try doing 10 victory push ups after each game you stomp, champ!", 
              "Welcome to the League of Draven! Nobody gets you down; the next game is yours to dominate. Keep those axes spinning by rotating those wrists to keep 'em loose!",
              "Shurima, your emperor is on a roll! Nothing restores glory like a reborn king, just like nothing restores a glow like lotion and face masks! Don't forget to practice skincare!",
              "People fear what they cannot understand. You AREN'T tilted? Stay that way by taking a few minutes between each game to mentally prepare!"
            ]},
            {"id": 2, "name": "Tilt Level 2", "tiltedness": 2, 'messages':[
              "Alright champ, looks like you are slaying it. Odds are, you aren't tilted at all - in fact you are probably feeling great! Remember to drink at least 8 cups of water a day to make sure your organs are healthy so you can keep stomping!",
              "Everybody dies, some just need a little help. Seems like you just had a little bit of bad luck! Give your hands & wrists a break (and avoid carpal tunnel) so your reflexes can be faster next time!"
              "Sometimes a shark takes the bait, and sinks the whole ship. Looks like you might've hooked a loss, but shake it off and reel in a win next time! Walk a lap around your room to prep!",
              "Master yourself, master the enemy. Seems like you've got that down! Look away from your screen to give your eyes a break, don't wanna become blind like Lee Sin! "
            ]},
            {"id": 3, "name": "Tilt Level 3", "tiltedness": 3, 'messages':[
              "Let's take a breather. There's no need to queue up again right away. Try burning away some of that frustration by doing 30 minutes of exercise - it will help clear your mind and help you win your next few games!",
              "So you had a couple rough games. Maybe even encountered some rough in-game bugs. Try talking a short walk outside to get some sunlight and see friendly bugs instead!"
              "Rules were made to be broken. Like buildings. Or people! Seems like the tilt is starting to get to you. Take a break and listen to some calming music!"
              "ok...like Rammus you are just rolling along, with a few unsavory bumps. Drink tons of water and eat enough to keep you fueled for a win!"
            ]},
            {"id": 4, "name": "Tilt Level 4", "tiltedness": 4, 'messages':[
              "Looks like you aren't having a great time. We predict that you are PRETTY tilted. Let's take a step away and take out some of that frustration doing some pushups.",
              "You're not invited to Tibber's tea party . . . or the winning party either. The tilt is evident, considering taking a break and catch up with friends",
              "FEEDING TIME! Like Fizz, you might be jumping in too much! Hunger never helps either, eat a snack to refuel that mana!",
              "Gems are truly, truly, truly outrageous. Especially tilted ones! Get your shine back by taking a shower to freshen up and wake up those muscles!"
              "1, 2, 3, 4! While 4 is a beautiful number, 4 is not the level of tilt you want to be at! Give your eyes a break so you don't miss your shots!"
            ]},
            {"id": 5, "name": "Tilt Level 5", "tiltedness": 5, 'messages':[
              "STOP. You are mega tilted. We predict that you are not about to have a fun time if you queue up again. Consider walking away from the game and doing some exercise to burn off some steam.",
              "Hmm, fortune doesn't favor fools (or the tilted). Today might not be your day, but you'll get 'em next time! Take the rest of the day off and come back to the rift tomorrow!",
              "Ally, enemy, I don't care. . ., they're all tilted! You're tilted so much you can't even be an ally to your own team. Take time off the rift for some much needed sleep",
              "I wonder what the ducks are plotting today. . . be like Ivern and go enjoy some nature! You're tilt is too high for your games to be productive, so take a long break!"
              ]},
            ]

tilt = ["1", "2", "3", "4", "5"]

api = Flask(__name__)

key = os.environ.get('RIOT_API_KEY', None)
print(key)
region = 'na1'
watcher = LolWatcher(key)

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
  
  @after_this_request
  def delete_static_files(response):
    files = glob.glob('./static/images/*.jpg', recursive=True)

    for f in files:
      try:
          os.remove(f)
      except OSError as e:
          print("Error: %s : %s" % (f, e.strerror))
    return response

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
  graph_get_history(summoner_name) # begin generating graphs
  player_data = watcher.summoner.by_name(region, summoner_name)
  user_data = get_data(player_data['id'], player_data['accountId'], watcher)
  user_df = pd.DataFrame(user_data)
  processed_df = postprocess(user_df)
  pred_x = np.array(processed_df)
  
  prediction = loaded_model.predict_proba(pred_x)
  print('prediction = ', prediction)
  if prediction is None:
    return "Your tilt-score could not be calculated because you haven't played any games yet today!"
  return round(prediction[0][1], 2)

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
    # fix win_g1 # 
    win_g0 = df['win_g0'].tolist()
    df['win_g0'] = fix_bools(win_g0)
    
    win_g1 = df['win_g1'].tolist()
    df['win_g1'] = fix_bools(win_g1)
    
    win_g2 = df['win_g2'].tolist()
    df['win_g2'] = fix_bools(win_g2)
    
    win_g3 = df['win_g3'].tolist()
    df['win_g3'] = fix_bools(win_g3)
    
    win_g4 = df['win_g4'].tolist()
    df['win_g4'] = fix_bools(win_g4)
    
    return df

def get_data(pid, accountid, watcher):
    '''
    Given player idea, generate a single row of data
    '''
    region = 'na1'
    mapping_list = [] # stores all mappings
    
    player_info = watcher.league.by_summoner('na1', pid)
    solo_queue_rank_info = [x for x in player_info if x['queueType'] == 'RANKED_SOLO_5x5'][0]  # query 1

    # compute match features #
    matches = watcher.match.matchlist_by_account(region, accountid)
    mapping = {}
    

    for idx, game in enumerate(matches['matches']):
        if idx < 5:
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
                mapping[f'gold_earned_g{idx}'] = gold_earned
                mapping[f'wards_placed_g{idx}'] = wards_placed 
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
            return mapping_list

    return mapping_list


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
    kda.save(filename = f'./static/images/{summoner}_kda.jpg', height=3, width=5, units = 'in', dpi=300)
    tddtc = graph_tddtc(df)
    tddtc.save(filename = f'./static/images/{summoner}_tddtc.jpg', height=3, width=5, units = 'in', dpi=300)
    # mddtc = graph_mddtc(df)
    # mddtc.save(filename = f'./static/images/{summoner}_mddtc.jpg', height=3, width=5, units = 'in', dpi=1000)
    # pddtc = graph_pddtc(df)
    # pddtc.save(filename = f'./static/images/{summoner}_pddtc.jpg', height=3, width=5, units = 'in', dpi=1000)
    gepm = graph_gepm(df)
    gepm.save(filename = f'./static/images/{summoner}_gepm.jpg', height=3, width=5, units = 'in', dpi=300)
    tmk = graph_tmk(df)
    tmk.save(filename = f'./static/images/{summoner}_tmk.jpg', height=3, width=5, units = 'in', dpi=300)
    return 'Files Saved!'

def graph_kda(df):
    try:
      kda = ggplot(df) + aes(x='games_ago', y='kda') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + ylim(0,30) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('KDA') + ggtitle("KDA Over the Last 5 Games")
    except:
      pass
    return kda

def graph_tddtc(df):
    tddtc = ggplot(df) + aes(x='games_ago', y='TotalDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Total Damage Dealt To Champions \n Per Minute') + ggtitle("Total Damage Dealt To Champions Per Minute \n Over the Last 5 Games")
    return tddtc

# def graph_mddtc(df):
#     mddtc = ggplot(df) + aes(x='games_ago', y='MagicDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Magic Damage Dealt To Champions \n Per Minute') + ggtitle("Magic Damage Dealt To Champions Per Minute \n Over the Last 5 Games")   
#     return mddtc

# def graph_pddtc(df):
#     pddtc = ggplot(df) + aes(x='games_ago', y='PhysicalDamageDealtToChampionsPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Physical Damage Dealt To Champions \n Per Minute') + ggtitle("Physical Damage Dealt To Champions Per Minute \n Over the Last 5 Games")
#     return pddtc

def graph_gepm(df):
    gepm = ggplot(df) + aes(x='games_ago', y='goldEarnedPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Gold Earned Per Minute') + ggtitle("Gold Earned Per Minute \n Over the Last 5 Games")
    return gepm

def graph_tmk(df):
    tmk = ggplot(df) + aes(x='games_ago', y='totalMinionsKilledPerMinute') + geom_line(color="black", linetype='dashed') + geom_point(aes(color='lane', size=3), show_legend={'size': False}) + scale_x_reverse() + geom_smooth(method = "lm") + xlab('Games Ago') + ylab('Total Minions Killed \n Per Minute') + ggtitle("Total Minions Killed Per Minute \n Over the Last 5 Games")
    return tmk

@api.route('/img/<summoner_postfix>')
def send_img(summoner_postfix):
    return send_from_directory('./static/images', f'{summoner_postfix}.jpg') ## need to fix this

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    api.run(host='0.0.0.0', port=port)
    # api.run(debug=True, port=port)