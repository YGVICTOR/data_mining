import numpy as np
import pandas as pd
import re
import requests
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics


# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
    data_df = pd.read_csv(data_file, encoding='latin-1')
    return data_df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    return list(df['Sentiment'].unique())


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    return pd.DataFrame(df.groupby('Sentiment')['UserName'].count()).reset_index().sort_values(by=['UserName'],
                                                                                               ascending=False).iloc[1][
        'Sentiment']


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    tmp_df = pd.DataFrame(df.groupby(['TweetAt', 'Sentiment'])['UserName'].count()).reset_index()
    return tmp_df[tmp_df['Sentiment'] == 'Extremely Positive'].sort_values(by=['UserName'], ascending=False).iloc[0][
        'TweetAt']


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    pd.set_option('display.max_columns', None)
    lower_tweet = df.apply(lambda x: x['OriginalTweet'].lower(), axis=1)
    df['OriginalTweet'] = lower_tweet
    return df


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    with_out_non_alphabetic_chars_tweet = df.apply(lambda x:re.sub(r'[^a-zA-Z ]',' ',x['OriginalTweet']).strip(),axis=1)
    df['OriginalTweet'] = with_out_non_alphabetic_chars_tweet
    return df


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    remove_multiple_consecutive_whitespaces_tweet = df.apply(lambda x:re.sub(r' {2,}',' ',x['OriginalTweet']).strip(),axis=1)
    df['OriginalTweet'] = remove_multiple_consecutive_whitespaces_tweet
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    tokenized_tweet = df.apply(lambda x: x['OriginalTweet'].split(),axis=1)
    df['OriginalTweet'] = tokenized_tweet
    df['OriginalTweet'] = tokenized_tweet
    return df


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    temp_df = tdf.explode('OriginalTweet')
    return temp_df[~temp_df.OriginalTweet.isna()].shape[0]


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    return len(tdf.explode('OriginalTweet')['OriginalTweet'].unique())


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    tdf_copy = pd.DataFrame(tdf['OriginalTweet'].copy(deep=True),columns=['OriginalTweet'])
    tdf_explode = tdf_copy.explode('OriginalTweet')
    tdf_explode['qty'] = 1
    return list(pd.DataFrame(tdf_explode.groupby("OriginalTweet")['qty'].count()).reset_index().sort_values(by=['qty'],ascending=False).reset_index(drop=True).iloc[:k]['OriginalTweet'])


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    def get_list_after_removing_stop_words(words,stop_words):
        return [x for x in words if x not in stop_words and len(x)>2]
    stopwords = requests.get(
        'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').content.decode(
        'utf-8').split("\n")
    tdf['OriginalTweet']= tdf.apply(lambda x : get_list_after_removing_stop_words(x['OriginalTweet'],stopwords),axis=1)
    return tdf


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    def porter_stemmer(words_list):
        ps = PorterStemmer()
        porter_stemmer_result = []
        # stem every word in news_words
        for w in words_list:
            porter_stemmer_result.append(ps.stem(w))
        return porter_stemmer_result
    tdf['OriginalTweet']= tdf.apply(lambda x : porter_stemmer(x['OriginalTweet']),axis=1)
    return tdf


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    # construct a sparse representation of the term-document matrix.
    vec = CountVectorizer()
    X = vec.fit_transform(df.OriginalTweet.to_numpy())
    y = df.Sentiment
    clf = MultinomialNB()
    clf.fit(X,y)
    y_hat = clf.predict(X)
    return y_hat


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive')
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return round(np.sum(np.diag(cm))/np.sum(cm),3)