import pandas as pd

import coronavirus_tweets
if __name__ == '__main__':
    coronavirus_tweets_df = coronavirus_tweets.read_csv_3('../data/coronavirus_tweets.csv')
    # # # print(df)
    # # # print(coronavirus_tweets.get_sentiments(df))
    # # # print(len(coronavirus_tweets.get_sentiments(df)))
    # # coronavirus_tweets.lower_case(df)
    # # # print(df)
    # # pd.set_option('display.max_columns', None)
    # # coronavirus_tweets.remove_non_alphabetic_chars(df)
    # #
    # # coronavirus_tweets.remove_multiple_consecutive_whitespaces(df)
    # # # print(df)
    # # # print(df)
    # # coronavirus_tweets.tokenize(df)
    # # # print(coronavirus_tweets.count_words_with_repetitions(df))
    # # # print(df)
    # # # print(coronavirus_tweets.count_words_without_repetitions(df))
    # # coronavirus_tweets.remove_stop_words(df)
    # # print(coronavirus_tweets.stemming(df))
    # # print(coronavirus_tweets.count_words_without_repetitions(df))
    # # print(coronavirus_tweets.count_words_with_repetitions(df))
    # # print(df)
    #
    #
    #
    #
    # # print(coronavirus_tweets.second_most_popular_sentiment(df))
    # # print(coronavirus_tweets.date_most_popular_tweets(df))
    # predicted_label = coronavirus_tweets.mnb_predict(df)
    # print(type(predicted_label))
    # print(predicted_label)
    # print(coronavirus_tweets.mnb_accuracy(predicted_label,df['Sentiment']))

    # from template_files import coronavirus_tweets
    import time

    # coronavirus_tweets_df = coronavirus_tweets.read_csv_3("coronavirus_tweets.csv")
    # #
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
    remove_multiple_consecutive_whitespaces_df = coronavirus_tweets.remove_multiple_consecutive_whitespaces(
        remove_non_alphabetic_chars_df)
    print(remove_multiple_consecutive_whitespaces_df["OriginalTweet"])
    tokenize_df = coronavirus_tweets.tokenize(remove_multiple_consecutive_whitespaces_df)
    print(tokenize_df["OriginalTweet_tokenized"])
    counts = coronavirus_tweets.count_words_with_repetitions(tokenize_df)
    print(counts)
    counts_without = coronavirus_tweets.count_words_without_repetitions(tokenize_df)
    print(counts_without)

    top_words = coronavirus_tweets.frequent_words(tokenize_df, 10)
    print("before filtering")
    print(top_words)

    tdf = coronavirus_tweets.remove_stop_words(tokenize_df)
    print(tdf.OriginalTweet_tokenized)
    counts = coronavirus_tweets.count_words_with_repetitions(tdf)
    print(counts)
    tdf = coronavirus_tweets.stemming(tdf)
    print(tdf.OriginalTweet_tokenized)
    counts = coronavirus_tweets.count_words_with_repetitions(tdf)
    print(counts)
    print("after filtering")
    top_words = coronavirus_tweets.frequent_words(tdf, 10)
    print(top_words)
    #
    print("begin")
    start_time = time.time()

    coronavirus_tweets_df = coronavirus_tweets.read_csv_3('../data/coronavirus_tweets.csv')
    y_hat = coronavirus_tweets.mnb_predict(coronavirus_tweets_df)
    print(time.time() - start_time)
    y_true = list(coronavirus_tweets_df.Sentiment.values)

    accuracy = coronavirus_tweets.mnb_accuracy(y_hat, y_true)
    print(accuracy)
