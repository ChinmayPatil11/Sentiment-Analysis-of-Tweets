import tweepy
from tweepy import OAuthHandler, API, Cursor
import twitter_credentials
import pandas as pd
import time

auth = OAuthHandler(twitter_credentials.api_key,twitter_credentials.api_secret)
auth.set_access_token(twitter_credentials.OAuth_token,twitter_credentials.OAuth_secret)

api = API(auth, wait_on_rate_limit = True)

def get_related_tweets(query):

    tweets_list = []
    count = 10
    try:
        for tweet in api.search(q='covid',lang='en',count=count):
            #print(tweet.text)
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})

        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,')
        time.sleep(3)
