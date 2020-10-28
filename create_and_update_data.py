# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 15 14:37:48 2020

@author: Ivan Martino

This script downloads the last tweets (about 3600) of a specific list of users from Twitter API.
It saves them in a specific file "all_tweets.txt" inside a given FOLDER.
The new file will look like:

all_tweets.txt
BOF
[[0, @screen_name, "This is the tweet", 10221237864237823 (tweet ID), "May 5, 1821" (tweet date)],
...
]
EOF

If you run it hours/days later, the script also upload the file with the new tweets.


++++++++++++++++++++
There are few important variables to set:
*The secret keys of the Twitter API
CONSUMER_KEY = '...'
CONSUMER_SECRET = '...'
ACCESS_TOKEN =  '...'
ACCESS_TOKEN_SECRET ='...'

**Then, also:
PATH, FILE_USERS, FILE_INFO, FOLDER, FILE_DATA_BASE.

PATH="..." contains the directory location, where to read and store the file.

FILE_USERS="..." is a text file where to store the screen_names of the Twitter users.
For example, the content of FILE_USERS="users.txt" could be

users.txt
BOF
@realDonaldTrump
@JoeBiden
@KamalaHarris
@Mike_Pence
EOF

FILE_INFO="..." this should contain the name of the file where to store the account information,
i.e. FILE_INFO = 'info'. Depending of the needs, this could be a txt-file or a pny-file.

FOLDER="..." is the name of the folder to place all information we get from TWITTER

FILE_DATA_BASE="..." is the root of the file names for the data we are creating
"""

#neeeded first in get_keys(FILE)
import os.path
import sys

#for the right folder and to save and load list
import os
import json

#neeeded first in initialize_api()
import twitter


#needed first to writes the tweets
import numpy as np

#VARIABLES
##string
PATH=""
FILE_USERS="users.txt"
LANGUAGES = ['en'] #this is the lenguage 'it'=Italian, 'en'=English
FOLDER="./dataset" #folder where to save the files of the tweets
FILE_DATA_BASE = 'tweet' #this is the root of the file names for the data we are creating
FILE_INFO = '' #summary file

#--
#CONSUMER_KEY = ''
#CONSUMER_SECRET = ''
#ACCESS_TOKEN =  ''
#ACCESS_TOKEN_SECRET =''


"""
We set the right path
"""
def setpath():
    os.chdir(PATH)
    return True

"""
We read the user screen names from FILE.
"""
def get_users(FILE):
    if os.path.isfile(FILE)==True:
        g=open(FILE, 'r' ,encoding='utf-8')
        NAMES = [l.strip('\n\r') for l in g.readlines()]
#        NAMES =[]
#        for line in g:
#            NAMES.append(line)
        g.close()
    else:
        print("Hej, there is no users to access API-TWITTER.\n Sorry we stop here!")
        sys.exit()

    a=-1;
    while a != False:
        a=clean(NAMES)
        if a!=False:
            NAMES.remove(a)

    print("We will analyze", len(NAMES), "twitter users.")
    return NAMES

"""
This function cleans a list of twitter users from double names.
"""
def clean(List):
    ''' Check if given list contains any duplicates '''
    S = set()
    for e in List:
        if e in S:
            return e
        else:
            S.add(e)
    return False

"""
The function returns the np.array
["@users", #ofTweetsGotten, #lasttweetsid]
"""
def get_info():
    f=FOLDER+"/"+FILE_INFO +".npy"
    if os.path.isfile(f)==True:
        return np.load(f)
    else:
        return []


"""
This function initializes the the twitter api using the keys.

It returns the twitter.Api
"""
def initialize_api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET):
    print("I inizialize twitter.api.\n")
    return twitter.Api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, tweet_mode='extended')



"""
This function gets as many tweets as possible from a user.
It uses the api.GetUserTimeline function that allows us to get at most 200 tweets per time
So we run it as many time as possible.

INPUT:
    api - the inizialize twitter api.
    Name - the screen name of a user, i.e. @scree_name
    N - it should be a number less than 3200, but greater than 200
        if it is greater than 3200, the function reaches for as many tweets as possible

It returns:
    tweets - the list of tweets
    tweets_id - the list of id of the tweets
    date - the list of date of tweets
"""
def get_tweets(api, Name, N):
    condition=True
    while condition:
        try:
            tweets #this is the list of tweets we are going to create
        except NameError:
            statuses=api.GetUserTimeline(screen_name=Name, count=200)
            tweets = [u.full_text for u in statuses] #we initialize "tweets"
            date=[u.created_at for u in statuses]
            tweets_id = [u.id for u in statuses]
        else:
            old_len=len(tweets_id)
            statuses=api.GetUserTimeline(screen_name=Name, count=200, max_id=(tweets_id[-1]-1))
            tweets.extend([u.full_text for u in statuses])
            date.extend([u.created_at for u in statuses])
            tweets_id.extend([u.id for u in statuses])
            if len(tweets)>N:
                condition=False
            if len(tweets)==old_len:
                condition=False
        del(statuses)
    return tweets, tweets_id, date


"""
This function saves the main list all_tweets in a txt-file with the same name.
It also saves the global information in FILE_INFO.npy
"""
def save_data(all_tweets, info_data):
    try:
        os.mkdir(FOLDER)
    except OSError:
        None #print ("Creation of the directory %s failed" % FOLDER)
    else:
        print ("We have created the directory %s to store the data" % FOLDER)

    with open(FOLDER+'/all_tweets.txt', 'w') as g:
        g.write(json.dumps(all_tweets))
    g.close

    f=FOLDER+"/"+FILE_INFO+".npy"
    np.save(f, np.array(info_data))
    return True



def load_data():
    with open(FOLDER+'/all_tweets.txt', 'r') as g:
        return json.loads(g.read())


"""
This function updates the tweet data in the library for a specific user.

INPUT:
    api - the inizialize twitter api.
    Name - the screen name of a user, i.e. @scree_name
    last_id - the id of the last tweet we have in the system

OUTPUT:
    new_tweets - the list of new tweets
    new_date - the list of date of new tweets
    new_tweets_id -  the id of the new tweets
"""
def update_data(api, Name, last_id):
    condition=True
    n=-1
    while condition:
        try:
            tweets #this is the list of tweets we are going to get from api
        except NameError:
            #print("Call api for for"+Name+"with ", last_id)
            statuses=api.GetUserTimeline(screen_name=Name, count=200)
            tweets = [u.full_text for u in statuses] #we initialize "tweets"
            date=[u.created_at for u in statuses]
            tweets_id = [u.id for u in statuses]
            #we check if among the new tweets there is the last tweet got
            if last_id in tweets_id:
                n=tweets_id.index(last_id)
                condition=False
        else:
            old_len=len(tweets_id)
            #print("Call api for "+Name+"with ", last_id)
            statuses=api.GetUserTimeline(screen_name=Name, count=200, max_id=(tweets_id[-1]-1))
            tweets.extend([u.full_text for u in statuses])
            date.extend([u.created_at for u in statuses])
            tweets_id.extend([u.id for u in statuses])
            #we check if among the new tweets there is the last tweet got
            if last_id in tweets_id:
                n=tweets_id.index(last_id, -200)#we check only the last 200
                condition=False

            if len(tweets)==old_len:
                condition=False

        del(statuses)
    if n==-1:
        n=len(tweets)
    return tweets[0:n], tweets_id[0:n], date[0:n]


"""
This is the main :)

The script get a list of users name from FILE_USERS and check for repetitions.
Then, using the twitter.API, we download the most recents tweets form Twitter,
up to the last tweet already saves (if this has been already run).

"""
def main():
    print("Hello,\nhere we create the up to date database of tweets from a list of users.\n")
    setpath()

    #we get the keys from the FILE_KEYS
    keys_file = open("keys.txt")
    lines = keys_file.readlines()
    CONSUMER_KEY = lines[0].rstrip()
    CONSUMER_SECRET = lines[1].rstrip()
    ACCESS_TOKEN = lines[2].rstrip()
    ACCESS_TOKEN_SECRET = lines[3].rstrip()
    
    #we initialize the api
    api = initialize_api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


    #get the users
    screen_names=get_users(FILE_USERS);
    x=clean(screen_names)
    if x!=False:
        print("The list has multiple elements; delete "+x)

    info_data=get_info()
    if len(info_data)==0:
        all_tweets=[]#this list will store all tweets information
        print("We will download about 3600 max for the following users:")
        for sn in screen_names:
            print(sn, "with label", screen_names.index(sn))
        print("\n")
        info_data=[]; #in this list, I append vectors like ["@users", #ofTweetsGotten, #lasttweetsid]
        for sn in screen_names:
            s = get_tweets(api, sn, 3600) #any number larger than 3200 will do.
            info_data.append([sn, len(s[0]), s[1][0]])
            print("The API has got", len(s[0]), "tweets from", sn)
            for i in range(len(s[0])):
                all_tweets.append([screen_names.index(sn), sn, s[0][i], s[1][i], s[2][i]])
    else:
        print("We will update the data we have.")
        print(info_data)
        update_info_data=[]
        all_tweets=load_data()
        print("We will update our database.")
        print("\n")
        for sn in screen_names:
            s = update_data(api, sn, int(info_data[screen_names.index(sn),2]))
            if len(s[0])>0:
                total=len(s[0])+int(info_data[screen_names.index(sn),1])
                update_info_data.append([sn, total, s[1][0]])
                print("The API has got", len(s[0]), "new tweets from", sn, "for a total of", total)
                for i in range(len(s[0])):
                    all_tweets.append([screen_names.index(sn), sn, s[0][i], s[1][i], s[2][i]])
            else:
                print("The API has got no new tweets for", sn)
                update_info_data.append([info_data[screen_names.index(sn), 0], info_data[screen_names.index(sn), 1], info_data[screen_names.index(sn), 2]])
        info_data=update_info_data

    print("\n")
    if save_data(all_tweets, info_data)==True:
        print("Data are saved")


if __name__ == '__main__':
    main()
