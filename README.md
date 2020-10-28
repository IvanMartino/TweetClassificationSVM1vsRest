# Twitter User Classification via SVM - one versus rest
Tweets Classification with SUPPORT VECTOR MACHINE, one versus rest
This is a quick and dirty script and it does what it should do.

This repository contains two files:
  - create_and_update_data.py 
    This file creates and uploads a database of tweets from a list of users. 
    The informations are stored in a unique file all_tweets.txt and in a summary file (.pny)
    
  - 9_tweets_classification_SUPPORT_VECTOR_MACHINE_1vsr.py
    Using the data created by create_and_update_data.py, this script classify the Twitter users via the simplest possible SVM.
    
#### Warning.
Accuracy is good when users are less than 4, decents for a dozen of accounts, but it drops down for larger sets of accounts. 

#### Training set cardinality.
Unfortunately, SVM gets super slow when the number of training example is larger than 15000, so we randomly select about 10000 elements from the traning set.
