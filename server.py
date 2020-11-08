from flask import Flask, json

tiltedness = [
            {"id": 1, "name": "Tilt Level One", "tiltedness": 1},
            {"id": 2, "name": "Tilt Level Two", "tiltedness": 2},
            {"id": 3, "name": "Tilt Level Three", "tiltedness": 3},
            {"id": 4, "name": "Tilt Level Four", "tiltedness": 4},
            {"id": 5, "name": "Tilt Level Five", "tiltedness": 5},
            ]

api = Flask(__name__)

@api.route('/companies', methods=['GET'])
def get_companies():
  return json.dumps(companies)

@api.route('/', methods=['GET'])
def get_home():
  return "hello :)"


if __name__ == '__main__':
    api.run()