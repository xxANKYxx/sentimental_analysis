from flask import Flask,render_template,flash,request,url_for,redirect,session
import numpy as np
from numpy import array
import re,os,random
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model


Img_folder = os.path.join('static','imges')
app = Flask(__name__)
app.config['UPLOAD_FOLDER']=Img_folder

def init():
    global  model,graph
    model = load_model('sentiment_analysis_model_new.h5')

    graph =tf.compat.v1.get_default_graph()







##########################################

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/predictions',methods=['POST','GET'])
def sentimental_prediction():
    text = request.form['text']
    Sentiment = ''
    max_review_length = 500
    word_to_id = imdb.get_word_index()
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    text = text.lower().replace("<br />", " ")
    text = re.sub(strip_special_chars, "", text.lower())

    words = text.split()  # split string into a list
    x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
    x_test = sequence.pad_sequences(x_test, maxlen=499)  # Should be same which you used for training data
    vector = np.array([x_test.flatten()])

    probab = model.predict(array([vector][0]))[0][0]
    emote = model.predict_classes(array([vector][0]))[0][0]
    if  emote  ==0:
        sentiment='Negative'
        img_filename=os.path.join(app.config['UPLOAD_FOLDER'],'sad1.jpg')
    else:
        sentiment = 'Positive'
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy1.jpg')
    return render_template('home.html', text=text,probability=probab ,sentiment=sentiment, image=img_filename)
###################
if __name__ =="__main__":
    app.debug = True
    init()
    app.run()