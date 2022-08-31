import adult
import pandas as pd
from sklearn import tree

if __name__ == '__main__':
    df = adult.read_csv_1('../data/adult.csv')

    # print(df)
    # print(adult.num_rows(df))
    # print(adult.column_names(df))
    # print(adult.missing_values(df))
    # print(adult.columns_with_missing_values(df))
    # print(adult.bachelors_masters_percentage(df))
    # print(adult.columns_with_missing_values(df))
    # print(adult.num_rows(df))

    data_frame_without_missing_values = adult.data_frame_without_missing_values(df)
    print("hhhh")
    print(df[df.isnull()])

    # print(adult.num_rows(data_frame_without_missing_values))
    # print(adult.columns_with_missing_values(data_frame_without_missing_values))
    # print(adult.one_hot_encoding(data_frame_without_missing_values))

    y_train = adult.label_encoding(data_frame_without_missing_values)
    x_train = adult.one_hot_encoding(data_frame_without_missing_values)
    print(x_train.shape)
    print(data_frame_without_missing_values.shape)
    y_hat = adult.dt_predict(x_train,y_train)
    print(adult.dt_error_rate(y_hat,y_train))

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf.fit(x_train.iloc[0:40000],y_train.iloc[0:40000])
    # y_hat = clf.predict(x_train.iloc[40000:])
    # y_true = y_train.iloc[40000:].copy().reset_index(drop=True)
    # print(adult.dt_error_rate(pd.Series(y_hat), pd.Series(y_true)))

    # error_rate = adult.accurate_error_estimation(x_train.iloc[0:6], y_train.iloc[0:6], num_fold=3)
    #
    for num_fold in range(2,10):
        error_rate = adult.accurate_error_estimation(x_train, y_train, num_fold=num_fold)
        print('error_rate:{} when num_fold = {}'.format(error_rate,num_fold))



