from flask import Flask, render_template, request, jsonify, after_this_request
import numpy as np
import pandas as pd
import sklearn
import pickle

path = 'Data-science/AI/machine-learning/ml-projects/weather-prediction/weather_model_2.pkl'
model_2 = pickle.load(open(path, 'rb'))
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

@app.route('/prediction', methods=['POST'])
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

    date_t = pd.Timestamp(data[5], unit='s', tz='Etc/GMT+8')
    date_t = pd.DatetimeIndex([date_t])
    month = date_t.month[0]
    week = date_t.week[0]
    day = date_t.day[0]
    hour = date_t.hour[0]

    pred_2 = model_2.predict([[humidity, pressure, temp, deg, speed, month, week, day, hour]])

    return jsonify({'pred': targets[pred_2[0]]})



if __name__ == "__main__":
    app.run()
