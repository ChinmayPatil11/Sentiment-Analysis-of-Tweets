import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from tensorflow import keras
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json


def clean_data(text):
    stop_words = set(stopwords.words('english'))
    punctuations_list = punctuation
    translator = str.maketrans('', '', punctuations_list)
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'@\S+', '', text) #usernames
    text = re.sub(r'#\S+', '', text) #hashtags
    text = re.sub(r'[a-zA-Z0-9._]+@[a-zA-Z]+\.(com|edu|net)', '', text) #emails
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(\S+\.com))', ' ', text) #urls
    text = text.translate(translator) #punctuations
    text = " ".join([word for word in str(text).split() if word not in stop_words]) #stopwords
    text = re.sub(r'\d+', '', text) #numbers
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split()) #lemmatize
    return text


def tokenize(df):
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    text_sequences = tokenizer.texts_to_sequences(df)
    text_sequences = pad_sequences(text_sequences, 200)
    return text_sequences


def make_prediction(predictions):
    pred_list = []
    for prediction in predictions:
        if prediction > 0.5:
            pred_list.append(1)
        else:
            pred_list.append(0)
    return pred_list