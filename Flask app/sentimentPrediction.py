from keras.models import load_model
from keras.preprocessing import sequence
import pickle as pkl

feature_vector = pkl.load(open(r'data\tokenizer.pkl', 'rb'))
model_cnn = load_model(r'data\cnn_weights-improvement-02-0.86.hdf5')
model_lstm = load_model(r'data\lstm_weights-improvement-98-0.57.hdf5')
log_reg = pkl.load(open(r'data\log_reg_0.6077_2020-09-27 10_30_41.459750.pkl', 'rb'))
textblob = pkl.load(open(r'data\textblob.pkl', 'rb'))

model_cnn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_lstm.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def get_polar(text):
    return 'Positive' if textblob(text).sentiment.polarity >= 0.4 else 'Negative'

def get_cnn_prediction(text):
    global feature_vector
    features = feature_vector.texts_to_matrix(text, mode='tfidf')
    prediction = model_cnn.predict_classes(features)
    return 'Positive' if prediction == 1 else 'Negative'

def get_lstm_prediction(text):
    global feature_vector
    features = feature_vector.texts_to_matrix(text, mode='tfidf')
    max_words = 500
    features = sequence.pad_sequences(features, maxlen=max_words)
    prediction = model_lstm.predict_classes(features)
    return 'Positive' if prediction == 1 else 'Negative'

def get_log_reg(text):
    global feature_vector
    features = feature_vector.texts_to_matrix(text, mode='tfidf')
    max_words = 500
    features = sequence.pad_sequences(features, maxlen=max_words)
    prediction = log_reg.predict(features)
    return 'Positive' if prediction == 1 else 'Negative'


def predict_text(text):
	resp = [get_polar(text), get_log_reg([text]), get_cnn_prediction([text]), get_lstm_prediction([text])]
	return ' '.join(resp)

# text = 'I agree that this is a great product and I am using it since past two weeks and it is really amazing. I fell in love with this product.'
# print('get_polar : ', get_polar(text))
# print('get_log_reg', get_log_reg([text]))
# print('get_cnn_prediction', get_cnn_prediction([text]))
# print('get_lstm_prediction', get_lstm_prediction([text]))
