from flask import Flask, request
import speech_recognition as sr
import pickle, joblib
from gensim_embeddings import convert_txt_vctr
import torch
from dl_models import ANN, CNN, LSTM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading the label_encoder
le = joblib.load("models/label_encoder.joblib")

app = Flask(__name__)

def get_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as audio_src:
        audio = recognizer.record(audio_src)
        text = recognizer.recognize_google(audio)

    return text

@app.route('/upload/rf', methods=['POST'])
def rf_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    rf_model = pickle.load(open("models/rf/model.pkl", "rb"))
    predict = rf_model.predict([emd_text])
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

@app.route('/upload/logistic', methods=['POST'])
def logistic_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    lr_model = pickle.load(open("models/lr/model.pkl", "rb"))
    predict = lr_model.predict([emd_text])
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

@app.route('/upload/xgboost', methods=['POST'])
def xgb_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    xgb_model = pickle.load(open("models/xgboost/model.pkl", "rb"))
    predict = xgb_model.predict([emd_text])
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

@app.route('/upload/ann', methods=['POST'])
def ann_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    input_dim = emd_text.shape[0]
    hidden_dim = 120
    output_dim = len(le.classes_)

    model = ANN(input_dim, hidden_dim, output_dim).to(device)

    model.load_state_dict(torch.load("models/ann/ann.pth", map_location=torch.device('cpu')))
    model.eval()
    output = model(torch.tensor([emd_text], dtype=torch.float32))
    _, predict = torch.max(output.data, 1)
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

@app.route('/upload/cnn', methods=['POST'])
def cnn_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    num_filters = 100  # Number of filters
    filter_sizes = [3, 4, 5]  # Filter sizes for convolutional layers
    input_dim = 1
    output_dim = len(le.classes_)
    model = CNN(input_dim=input_dim, num_filters=num_filters, filter_sizes=filter_sizes, output_dim=output_dim,
                dropout=0.25).to(device)

    model.load_state_dict(torch.load("models/cnn/cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    output = model(torch.tensor([emd_text], dtype=torch.float32))
    _, predict = torch.max(output.data, 1)
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

@app.route('/upload/lstm', methods=['POST'])
def lstm_upload():
    # Check if the request contains a file
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400

    audio_file = request.files['audio']
    text = get_text_from_audio(audio_file=audio_file)
    emd_text = convert_txt_vctr(text.split())

    input_size = emd_text.shape[0]
    hidden_size = 256
    num_layers = 2
    output_dim = len(le.classes_)
    model = LSTM(input_size=input_size, hidden_size=hidden_size,
                 output_size=output_dim, num_layers=num_layers).to(device)

    model.load_state_dict(torch.load("models/lstm/lstm.pth", map_location=torch.device('cpu')))
    model.eval()
    output = model(torch.tensor([emd_text], dtype=torch.float32))
    _, predict = torch.max(output.data, 1)
    predict_label = le.inverse_transform(predict)

    return {
        "text": text,
        "predicted_product": predict_label[0]
    }

if __name__ == '__main__':
    app.run()
