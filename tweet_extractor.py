import tweepy
from tweepy import OAuthHandler, API
import twitter_credentials
import pandas as pd
import time

auth = OAuthHandler(twitter_credentials.api_key, twitter_credentials.api_secret)
auth.set_access_token(twitter_credentials.OAuth_token, twitter_credentials.OAuth_secret)

api = API(auth, wait_on_rate_limit=True)


def get_trends_on_location(id):
    trends = api.trends_place(id)
    trend_list = []
    for value in trends:
        for trend in value['trends']:
            trend_list.append((trend['name']))
    return trend_list[:5]


def get_related_tweets(query):
    tweets_list = []
    query = query + '-filter:retweets'
    count = 10
    try:
        for tweet in api.search(q=query, lang='en', count=count, geocode='19.076090,72.877426,15km'):
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})

        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', e)
        time.sleep(3)
