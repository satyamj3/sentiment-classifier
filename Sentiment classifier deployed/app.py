# app.py
from flask import Flask
import numpy as np
from flask import render_template
from sentimentPrediction import predict_text
# from sentimentPrediction import get_log_reg
# from sentimentPrediction import get_cnn_prediction
# from sentimentPrediction import get_lstm_prediction

app = Flask(__name__)

def hello():
	return 'Hello world'

@app.route("/")
def home():
	return render_template('homepage.html')

@app.route('/response/<string:text>', methods=['GET','POST'])
def response(text):
	# response_list = []
	# response_list.append(get_polar([text]))
	# response_list.append(get_log_reg(text))
	# response_list.append(get_cnn_prediction(text))
	# response_list.append(get_lstm_prediction(text))
	# print('Response List : ', response_list)
	# res = np.array(["Positive", "Positive", "Positive", "Positive"])
	# return text
	return predict_text(text)

if __name__ == "__main__":
	app.run(debug=False)
	
	
# url: http://127.0.0.1:5000/