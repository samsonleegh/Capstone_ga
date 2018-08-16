import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
from keras.models import load_model
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model1 = load_model('toxic_md1_preemb4.h5')
model2 = load_model('toxic_md2_preemb4.h5')
def text_clean(df, text_col):
    df[text_col] = df[text_col].apply(lambda x: re.sub(',', '', x)) #replace commas without space to prevent additional spaces in sentences
    df[text_col] = df[text_col].apply(lambda x: re.sub("'", '', x)) #replace ' next to get words like dont, wont, etc.
    df[text_col] = df[text_col].apply(lambda x: re.sub('[^ a-zA-Z0-9!]', ' ', x).lower()) #replace weird char without ! (!+  seems to appear quite abit in toxic comments) + lower case
    df[text_col] = df[text_col].apply(lambda x: " ".join(x.split())) #remove additional spaces

#-------- ROUTES GO HERE -----------#
@app.route('/page')
def page():
   with open("Toxic Detector.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result', methods=['POST'])
def result():
    '''Gets prediction using the HTML form'''
    # if flask.request.method == 'POST':
    inputs = flask.request.form
    input1 = inputs['comment']
    input_df = pd.DataFrame()
    input_df = pd.DataFrame([input1], columns=['comment_text'])
    text_clean(input_df, 'comment_text')
    input_seq = tokenizer.texts_to_sequences(input_df['comment_text'])
    pad_seq = pad_sequences(input_seq, maxlen=150)
    y_pred1 = model1.predict(pad_seq)
    y_pred2 = model2.predict(pad_seq)
    neg = round(y_pred1[0][0]*100,2)
    tox = round(y_pred2[0][0]*100,2)
    s_tox = round(y_pred2[0][1]*100,2)
    obs = round(y_pred2[0][2]*100,2)
    thr = round(y_pred2[0][3]*100,2)
    ins = round(y_pred2[0][4]*100,2)
    id_h = round(y_pred2[0][5]*100,2)
    if y_pred1[0][0]<0.5:
        returnString2 = '<p>Negativity: none detected'
        return returnString2
    else:
        returnString2 = '<p>Negativity: {0}%</p><p>Toxic: {1}%</p><p>Severe Toxic: {2}%</p><p>Obscene: {3}%</p><p>Threat: {4}%</p><p>Insult: {5}%</p><p>Identity Hate: {6}%</p>'.format(neg,tox,s_tox,obs,thr,ins,id_h)
        return returnString2


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000

    app.run(HOST, PORT)
