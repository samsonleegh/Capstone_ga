import flask
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
#haiku model
from keras.models import load_model
from pickle import load
import numpy as np
import pandas as pd
import random, sys
import re
#load the model
model = load_model('haiku_30.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
inv_map = {v: k for k, v in mapping.items()}
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(a))
    return np.random.choice(choices, p=dist)#random.choices to implement weighted random selection

#spelling check without removing punctuations for master output
from nltk.tokenize import TweetTokenizer
from autocorrect import spell
def cleaning(sentence):
    tokenizer_words = TweetTokenizer()
    spellcheck = []
    for word in tokenizer_words.tokenize(sentence):
        if word not in [',','!','.',':']:
            spellcheck.append(spell(word))
        else:
            spellcheck.append(word)
    corr_sent = ' '.join(spellcheck)
    return corr_sent

#zenmaster model
import markovify
import spacy
import numpy as np
import json
from autocorrect import spell
with open('zen_phy.json', 'r') as f:
    model_json = json.load(f)
three_model = markovify.Text.from_json(model_json)

#-------- ROUTES GO HERE -----------#
@app.route('/haiku')
def page():
    with open("haiku.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/master')
def page2():
    with open("master.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/result1', methods=['POST'])
def result():
    '''Gets prediction using the HTML form'''
    # if flask.request.method == 'POST':
    inputs = flask.request.form
    usr_input = inputs['comment']
    usr_input = re.sub('[^a-zA-Z0-9 \n\.]', '', usr_input).lower()
    sentence = ('{:>' + str(20) + '}').format(usr_input[:20]).lower()
    generated = ''
    generated += sentence
    # sys.stdout.write(generated)
    tot_lines = 0
    tot_chars = 0
    while True:
        if tot_lines > 3 or tot_chars > 120:
            break
        x = np.zeros((1, 20, len(mapping)))
        for t, char in enumerate(sentence):
            x[0, t, mapping[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.5)
        next_char = inv_map[next_index]

        tot_chars += 1
        generated += next_char
        if next_char == '\n':
            tot_lines += 1
        sentence = sentence[1:] + next_char
        # sys.stdout.write(next_char)
        # sys.stdout.flush()
    generated = generated.replace('\n\n','<br>')
    generated = generated.replace('\n','<br>')
    return generated

@app.route('/result2', methods=['POST'])
def result2():
    '''Gets prediction using the HTML form'''
    # if flask.request.method == 'POST':
    inputs = flask.request.form
    usr_input = inputs['comment2']
    # usr_input2 = spell(usr_input)
    # try:
    #     gen_1 = '<b>Yours</b>: '+ three_model.make_sentence_with_start(usr_input2, strict=False)
    # except:
    #     gen_1 = 'None'
    # gen_2 = '<b>Zen Master</b>: ' + three_model.make_sentence() + ' ' + three_model.make_sentence()
    # gen_3 = gen_1 + '<br><br>' + gen_2

    usr_input2 = spell(usr_input)
    try:
        gen_1 = cleaning(three_model.make_sentence_with_start(usr_input2, strict=False))
        gen_1_clean = '<b>Yours</b>: ' + gen_1
    except:
        gen_1_clean = 'None'

    gen_2 = cleaning(three_model.make_sentence())

    gen_2_clean = '<b>Zen Master</b>: ' + gen_2
    gen_3 = gen_1_clean + '<br><br>' + gen_2_clean
    return gen_3

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4001

    app.run(HOST, PORT)
