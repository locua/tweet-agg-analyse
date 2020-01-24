import pandas as pd
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
# read csv and store text of tweets in pandas Series
boris = pd.read_csv('borisjohnson.csv')
jerem = pd.read_csv('jeremycorbyn.csv')
jeris = pd.read_csv('jeris.csv')
tweet_boris = boris['text']
tweet_jerem = jerem['text']
tweet_jeris = jeris['text']
# sentiment analyser class
analyser = SentimentIntensityAnalyzer()
# analyse tweet
#print(analyser.polarity_scores(tweet_text[0]))
# function returns list of compound sentiment from list of strings
def get_compound_sentiments(tweets):
    sents = []
    for i in tweets:
        sent = analyser.polarity_scores(str(i))
        sents.append(sent['compound'])
    return sents
# get compound sentiments and put in dataframe
tweet_sents_boris = get_compound_sentiments(tweet_boris)
tweet_sents_jerem = get_compound_sentiments(tweet_jerem)
tweet_sents_jeris = get_compound_sentiments(tweet_jeris)
bsents = pd.DataFrame.from_dict(tweet_sents_boris)
jsents = pd.DataFrame.from_dict(tweet_sents_jerem)
jbsents = pd.DataFrame.from_dict(tweet_sents_jeris)
# plots
ax1 = plt.axes()
jsents.hist(ax=ax1)
plt.title("jeremy")
bsents.hist()
plt.title("boris")
jbsents.hist()
plt.title("@borisjohnson @jeremycorbyn")
plt.show()

