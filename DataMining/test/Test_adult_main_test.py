import adult

adult_csv = adult.read_csv_1("../data/adult.csv")
# print(adult.num_rows(adult_csv), len(adult.column_names(adult_csv)), adult.missing_values(adult_csv),
#       adult.columns_with_missing_values(adult_csv), adult.bachelors_masters_percentage(adult_csv))
# print(adult.data_frame_without_missing_values(adult_csv).shape)
# print(adult_csv.shape)


# test
print(adult.num_rows(adult_csv))
print(adult.column_names(adult_csv))
test = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek', 'native-country', 'class']
print(test==adult.column_names(adult_csv))
print(adult.missing_values(adult_csv))
print(adult.columns_with_missing_values(adult_csv))
print(adult.bachelors_masters_percentage(adult_csv))
print(adult.data_frame_without_missing_values(adult_csv).shape)
with_out_missing_value = adult.data_frame_without_missing_values(adult_csv)
print(adult.one_hot_encoding(with_out_missing_value).shape)
print(adult.label_encoding(with_out_missing_value))
one_hot = adult.one_hot_encoding(with_out_missing_value)
label = adult.label_encoding(with_out_missing_value)
print(adult.dt_predict(one_hot,label))
label_hat = adult.dt_predict(one_hot,label)
print(label_hat)
error_rate = adult.dt_error_rate(label_hat, label)
print(error_rate)



# df_without_missing = adult.data_frame_without_missing_values(adult_csv)
# adult_one_hot = adult.one_hot_encoding(df_without_missing)
# # print(adult_one_hot)
# # print(df_without_missing)
# label = adult.label_encoding(df_without_missing)
# # print(label)
# label_hat = adult.dt_predict(adult_one_hot,label)
# # print(label_hat)
# error_rate = adult.dt_error_rate(label_hat, label)
# print(error_rate)
# # print(adult_one_hot)
# # adult_csv = adult_csv.dropna()
# # y_true = adult.label_encoding(adult_csv[["class"]])
# # y_predict = adult.dt_predict(adult_one_hot, y_true)
# # adult.dt_error_rate(y_predict, y_true)
