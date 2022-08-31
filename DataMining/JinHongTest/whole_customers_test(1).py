import wholesale_customers
import time

wc_csv = wholesale_customers.read_csv_2("../data/wholesale_customers.csv")
summaries = wholesale_customers.summary_statistics(wc_csv)
# print(summaries)
summary_df = wholesale_customers.summary_statistics(wc_csv)
print(summary_df)
standardize_df = wholesale_customers.standardize(wc_csv)
print(standardize_df)
print(wc_csv)
clusters = wholesale_customers.kmeans(wc_csv, 3)
print(clusters)
kmeans_score = wholesale_customers.clustering_score(wc_csv, clusters)
print(kmeans_score)
result = wholesale_customers.cluster_evaluation(wc_csv)
print(result)

wholesale_customers.scatter_plots(wc_csv)