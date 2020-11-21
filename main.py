from subprocess import run, PIPE

from flask import logging, Flask, render_template, request
import os
import numpy as np
import librosa
from joblib import load

model = load("modelo.joblib")
app = Flask(__name__)

def predict(nome_arq):

    X, sample_rate = librosa.load(os.path.join(nome_arq), res_type='kaiser_best')

    stft=np.abs(librosa.stft(X))
    atributos=np.array([])

    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    atributos=np.hstack((atributos, mfccs))

    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    atributos=np.hstack((atributos, chroma))

    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    atributos=np.hstack((atributos, mel))
    x = []
    x.append(atributos)
    resposta = model.predict(x)[0]
    return resposta

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio', methods=['POST'])
def audio():
    with open('audio.wav', 'wb') as f:
        f.write(request.data)
    previsao = str(predict('audio.wav'))
    print(previsao)
    return str(previsao)


if __name__ == "__main__":

    app.logger = logging.create_logger(app)
    app.run(debug=True)