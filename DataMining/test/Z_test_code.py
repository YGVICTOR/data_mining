import coronavirus_tweets as ct
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行


df = ct.read_csv_3("../data/coronavirus_tweets.csv")
y_pred = ct.mnb_predict(df)
print(ct.mnb_accuracy(y_pred,df["Sentiment"]))


sentiments = ct.get_sentiments(df)
second_most_popular_sentiment = ct.second_most_popular_sentiment(df)
print("second_most_popular_sentiment:")
print(second_most_popular_sentiment)
print("\n")

date_most_popular_tweets = ct.date_most_popular_tweets(df)
print("date_most_popular_tweets:")
print(date_most_popular_tweets)
print("\n")

ct.lower_case(df)
print("lower_case:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")


ct.remove_non_alphabetic_chars(df)
print("after remove_non_alphabetic_chars:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")

ct.remove_multiple_consecutive_whitespaces(df)
print("after remove_multiple_consecutive_whitespaces:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")

ct.tokenize(df)
print("after tokenize:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")

total_word = ct.count_words_with_repetitions(df)
print("total_word with repetition:")
print(total_word)
print("\n")


total_word = ct.count_words_without_repetitions(df)
print("total_word without repetition:")
print(total_word)
print("\n")
# 这段代码跑的太慢了


frequent_words = ct.frequent_words(df, 10)
print("frequent_words:")
print(frequent_words)
print("\n")

ct.remove_stop_words(df)
print("after remove_stop_words:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")

frequent_words = ct.frequent_words(df, 10)
print("frequent_words:")
print(frequent_words)
print("\n")


ct.stemming(df)
print("after stemming:")
for i in range(10):
    print(df["OriginalTweet"][i])
print("\n")






