import pandas as pd
import tensorflow as tf
from tensorflow import keras
from clean_data import clean_data, tokenize,make_prediction
from keras.models import load_model
from tweet_extractor import get_related_tweets
import spacy

model = load_model('classification_model.h5')


def find_entities(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    entity_list = []
    for ent in doc.ents:
        inner_list = []
        inner_list.append(ent.text)
        inner_list.append(ent.label_)
        entity_list.append(inner_list)
    return entity_list


def get_results(id):
    tweets = get_related_tweets(id)
    tweets['cleaned_text'] = tweets['tweet_text'].apply(lambda x: clean_data(x))
    tweets['length'] = tweets['cleaned_text'].apply(lambda x: len(x))
    tweets = tweets[tweets['length'] < 200]
    text_sequences_to_predict = tokenize(tweets['cleaned_text'])
    preds = model.predict(text_sequences_to_predict)
    preds = list(preds)
    preds = make_prediction(preds)
    tweets['prediction'] = preds
    tweets['prediction'] = tweets['prediction'].replace({0: 'negative', 1: 'positive'})
    tweets['entities'] = tweets['tweet_text'].apply(lambda x: find_entities(x))
    return tweets
