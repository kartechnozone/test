#!/usr/bin/python
import datetime
import tweepy
import unicodecsv #Import csv
from textblob import TextBlob
import time

consumer_key = 'Enter the consumer key'
consumer_secret = 'Enter the consumer_secret key'
access_token = 'Enter the access_token key'
access_token_secret = 'Enter the access_token_secret key'
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
index = 0
api = tweepy.API(auth)
while True:
    #enter the query below
    query = ' '
    moment = (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # Open/create a file to append data to
    csvFile = open(r'C:\Users\GL63\Desktop\mivsrr\tweets' +'-' + query + ' - ' + moment + '.csv'  , 'ab')

    #Use csv writer
    csvWriter = unicodecsv.writer(csvFile)

    csvWriter.writerow(['query = '+ query])
    csvWriter.writerow(['event = '+ moment])

    csvWriter.writerow(['Time','Id','user name','Text','Created at','Place','Retweet count',"Ploarity",'Subjectivity'])
    for tweet in tweepy.Cursor(api.search,
                            q = query,
                            lang = "en").items(100):

        # Write a row to the CSV file. I use encode UTF-8
        a = tweet.text.encode('utf-8')
        b = str(a.decode('utf-8', 'strict'))
        analysisPol = TextBlob(b).polarity
        analysisSub = TextBlob(b).subjectivity
        csvWriter.writerow([datetime.datetime.now().strftime("%Y-%m-%d  %H:%M"), tweet.id,tweet.user.screen_name,b, tweet.created_at,tweet.user.location if tweet.user.location else None,tweet.retweet_count,analysisPol,analysisSub])
        print (tweet.created_at, tweet.text)
    csvFile.close()
    index =index + 1
    time.sleep(180)