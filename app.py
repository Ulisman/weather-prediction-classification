from flask import Flask, render_template, request, jsonify, after_this_request
import numpy as np
import pandas as pd
import sklearn
import pickle

path = '/Users/ulisman/Library/Mobile Documents/com~apple~CloudDocs/python copy/Data-science/AI/machine-learning/ml-projects/weather-prediction/weather_model.pkl'
model = pickle.load(open(path, 'rb'))
targets = {
    0: 'Sky is clear',
    1: 'Light rain',
    2: 'Overcast clouds',
    3: 'Broken clouds',
    4: 'Mist',
    5: 'Scattered clouds',
    6: 'Few clouds',
    7: 'Moderate rain'
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    data = request.get_json()
    humidity = data[0]
    pressure = data[1]
    temp = data[2]
    deg = data[3]
    speed = data[4]

    date_t = pd.Timestamp(data[5], unit='s')
    date_t = pd.DatetimeIndex([date_t])
    month = date_t.month[0]
    week = date_t.week[0]
    day = date_t.day[0]
    hour = date_t.hour[0]

    pred = model.predict([[humidity, pressure, temp, deg, speed, month, week, day, hour]])

    print(pred)
    print(targets[pred[0]])

    return jsonify({'pred': targets[pred[0]]})





if __name__ == "__main__":
    app.run()