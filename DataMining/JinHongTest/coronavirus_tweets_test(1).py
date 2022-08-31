import coronavirus_tweets
import time
import numpy as np

# print(list(word_list))

start = time.time()
coronavirus_tweets_df = coronavirus_tweets.read_csv_3("../data/coronavirus_tweets.csv")
possible_sentiments = coronavirus_tweets.get_sentiments(coronavirus_tweets_df)
print(possible_sentiments)
second_most_popular_sentiment = coronavirus_tweets.second_most_popular_sentiment(coronavirus_tweets_df)
print(second_most_popular_sentiment)
date_most_popular_tweets = coronavirus_tweets.date_most_popular_tweets(coronavirus_tweets_df)
print(date_most_popular_tweets)
lower_case_df = coronavirus_tweets.lower_case(coronavirus_tweets_df)
print(lower_case_df["OriginalTweet"])
remove_non_alphabetic_chars_df = coronavirus_tweets.remove_non_alphabetic_chars(lower_case_df)
print(remove_non_alphabetic_chars_df["OriginalTweet"])
print(lower_case_df['OriginalTweet'])
remove_multiple_consecutive_whitespaces_df = coronavirus_tweets.remove_multiple_consecutive_whitespaces(
    remove_non_alphabetic_chars_df)
print(remove_multiple_consecutive_whitespaces_df["OriginalTweet"])
tokenize_df = coronavirus_tweets.tokenize(remove_multiple_consecutive_whitespaces_df)
print(tokenize_df["OriginalTweet"])
counts = coronavirus_tweets.count_words_with_repetitions(tokenize_df)
print(counts)
counts_without = coronavirus_tweets.count_words_without_repetitions(tokenize_df)
print(counts_without)

top_words = coronavirus_tweets.frequent_words(tokenize_df, 10)
print("before filtering")
print(top_words)

tdf = coronavirus_tweets.remove_stop_words(tokenize_df)
print(tdf.OriginalTweet)
counts = coronavirus_tweets.count_words_with_repetitions(tdf)
print(counts)
tdf = coronavirus_tweets.stemming(tdf)
print(tdf.OriginalTweet)
counts = coronavirus_tweets.count_words_with_repetitions(tdf)
print(counts)
print("after filtering")
top_words = coronavirus_tweets.frequent_words(tdf, 10)
print(top_words)

print("begin")
start_time = time.time()

coronavirus_tweets_df = coronavirus_tweets.read_csv_3("../data/coronavirus_tweets.csv")
y_hat = coronavirus_tweets.mnb_predict(coronavirus_tweets_df)
# print(time.time() - start_time)
y_true = list(coronavirus_tweets_df.Sentiment.values)

accuracy = coronavirus_tweets.mnb_accuracy(y_hat, y_true)
print(accuracy)
print(time.time() - start)
