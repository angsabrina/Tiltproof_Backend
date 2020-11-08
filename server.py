from flask import Flask, json
import random
import os

tiltedness = [
            {"id": 1, "name": "Tilt Level One", "tiltedness": 1},
            {"id": 2, "name": "Tilt Level Two", "tiltedness": 2},
            {"id": 3, "name": "Tilt Level Three", "tiltedness": 3},
            {"id": 4, "name": "Tilt Level Four", "tiltedness": 4},
            {"id": 5, "name": "Tilt Level Five", "tiltedness": 5},
            ]

tilt = ["1", "2", "3", "4", "5"]

api = Flask(__name__)

@api.route('/gettilt', methods=['GET'])
def get_tiltedness():
  return random.choice(tilt)

@api.route('/alltilts', methods=['GET'])
def get_alltilts():
  return json.dumps(tiltedness)

@api.route('/', methods=['GET'])
def get_home():
  return "hello :)"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    api.run(host='0.0.0.0', port=port)