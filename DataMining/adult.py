import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
    data_df = pd.read_csv(data_file)
    columns = column_names(data_df)
    if 'fnlwgt' in columns:
        return data_df.drop(['fnlwgt'], axis=1)
    else:
        return data_df


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    return df.shape[0]


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    return list(df.columns)


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    return sum(df.isnull().sum())


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    df_is_null = df.isnull().sum()
    name = list(df_is_null.index)
    result = [name[i] for i in range(len(df_is_null)) if df_is_null.iloc[i] != 0]
    return result


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    num_bachelor_master = df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')].shape[0]
    num_total = df.shape[0]
    return np.round(float(num_bachelor_master) / float(num_total), 3)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    return df.dropna(how='any', axis=0)


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
    df_types = list(df.dtypes.index)
    df_raw = df.drop('class', axis=1)
    df_quantitative_names = [df_types[i] for i in range(len(df_types)) if df.dtypes.iloc[i] != 'object']
    df_raw[df_quantitative_names] = df_raw[df_quantitative_names].astype(str)
    # Drop all instances with missing values
    new_df = df_raw.dropna()
    # creating instance of one-hot-encoder
    one_hot_df = pd.get_dummies(new_df[new_df.columns])
    return one_hot_df


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    # creating instance of label encoder
    new_df = df.dropna()
    le = LabelEncoder()
    le.fit(new_df['class'])
    return pd.Series(le.transform(new_df['class']))


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(X_train, y_train)
    # tree.export_graphviz(clf, out_file='tree.dot', impurity=True)
    y_hat = clf.predict(X_train)
    return pd.Series(y_hat)


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    M_train = y_pred.shape[0]
    # count the number of correctly predicted labels
    error = 0.0
    for i in range(M_train):
        if y_pred[i] != y_true[i]:
            error += 1
    error_rate = error / M_train
    return error_rate


# the following is for optional part
def __shuffle__(X_train, y_train):
    x = X_train.reset_index(drop=True)
    x['label'] = y_train
    X_train_shuff = shuffle(x, random_state=100)
    return X_train_shuff


def accurate_error_estimation(X_train, y_train, num_fold):
    error_rate = []
    # first shuffle the data frame
    X_train_shuff = __shuffle__(X_train, y_train).reset_index(drop=True)
    # apply n fold cross validation
    kf = KFold(n_splits=num_fold)
    # separate the feature and the response variables
    y = X_train_shuff['label']
    X_train_shuff = X_train_shuff.drop('label', axis=1)
    # apply n fold cross validation
    for train_index, test_index in kf.split(X_train_shuff):
        # check the index of training set and test set
        X_train, X_test = X_train_shuff.iloc[train_index], X_train_shuff.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # bulid the model
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        # training model on training set
        clf.fit(X_train, y_train)
        # test model on test set
        y_hat = pd.Series(clf.predict(X_test))
        y_true = pd.Series(y_test).reset_index(drop=True)
        # compute the score
        current_error_rate = dt_error_rate(y_hat, y_true)
        error_rate.append(current_error_rate)
    return np.average(np.array(error_rate))