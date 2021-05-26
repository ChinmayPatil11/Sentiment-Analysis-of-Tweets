import flask
from flask import Flask, render_template, request, url_for, redirect
from tweet_extractor import get_related_tweets
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from clean_data import clean, tokenize, make_predict, remove_usernames, remove_hashtags, remove_emails,remove_urls, remove_numbers, remove_punctuation, remove_stopwords, lemmatizer
from entity_finder import entity_recognizer

app = Flask(__name__)
model = load_model('classification_model.h5')
global tweets
global dictionary

def request_results(name):
    tweets = get_related_tweets(name)
    print(tweets['tweet_text'])
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_usernames(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_hashtags(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_emails(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_urls(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_punctuation(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_stopwords(x))
    tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: lemmatizer(x))
    tweets['length'] = tweets['tweet_text'].apply(lambda x:len(x))
    tweets = tweets[tweets['length'] < 200]
    text_sequences_to_predict = tokenize(tweets['tweet_text'])
    preds = model.predict(text_sequences_to_predict)
    preds = list(preds)
    preds = make_predict(preds)
    tweets['prediction'] = preds
    tweets['prediction'] = tweets['prediction'].replace({0:'negative',1:'positive'})

    return tweets

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def get_data(): #name
    if request.method == 'POST':
        user = request.form['text']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    tweets = request_results(name)
    output_tweets = tweets.drop(['created_at','tweet_id','length'],axis=1)
    dictionary = dict(zip(tweets['tweet_text'],tweets['prediction']))
    return render_template('success.html',lines=dictionary)

if __name__ == '__main__':
    app.run(debug=True)
