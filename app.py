import flask
from flask import Flask, render_template, request, url_for, redirect
import h5py
from tweet_extractor import get_related_tweets
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from clean_data import clean, tokenize, make_predict, remove_usernames, remove_hashtags, remove_emails,remove_urls, remove_numbers, remove_punctuation, remove_stopwords, lemmatizer


app = Flask(__name__)
#model = h5py.File('classification_model.h5','r+')
model = load_model('classification_model.h5')
global tweets

def request_results(name):
    tweets = get_related_tweets(name)
    #tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: clean(x))
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
    preds = model.predict(text_sequences_to_predict) #tweets['predictions']
    preds = make_predict(preds)
    tweets['prediction'] = preds
    tweets['prediction'] = tweets['prediction'].replace({0:'negative',1:'positive'})
    return (tweets)

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
    #no_of_tweets = tweets['prediction'].value_counts()
    #plt.bar(no_of_tweets.index, no_of_tweets)
    #plt.savefig('/static/images/plot.png')
    output_tweets = tweets.drop(['created_at','tweet_id','length'],axis=1)
    return render_template('success.html',tables=[output_tweets.to_html(classes='data')], titles=tweets.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
