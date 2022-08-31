import wholesale_customers
import time
wc_csv = wholesale_customers.read_csv_2("../data/wholesale_customers.csv")
summary_df = wholesale_customers.summary_statistics(wc_csv)
print(summary_df)
standardise_df = wholesale_customers.standardize(wc_csv)
print(standardise_df)
clusters = wholesale_customers.kmeans(wc_csv,3)
# print(clusters)
kmeans_score = wholesale_customers.clustering_score(wc_csv,clusters)
# print(kmeans_score)
result = wholesale_customers.cluster_evaluation(wc_csv)
print(result)
wholesale_customers.scatter_plots(wc_csv)

# summaries = wholesale_customers.summary_statistics(wc_csv)
# print(summaries)
# kmeans_result = wholesale_customers.kmeans(wc_csv, 2)
# print(kmeans_result)
# agglomerative_result = wholesale_customers.agglomerative(wc_csv, 2)
# print(agglomerative_result)
# k_means_score = wholesale_customers.clustering_score(wc_csv, kmeans_result)
# agglomerative_score = wholesale_customers.clustering_score(wc_csv, agglomerative_result)
# print(k_means_score)
# print(agglomerative_score)
# start_time = time.time()
# result = wholesale_customers.cluster_evaluation(wc_csv)
# print(result)
# print(time.time() - start_time)
# scores = result["Silhouette Score"]
# print(max(scores))
# # wholesale_customers.scatter_plots(wc_csv)
# standardized_df = wholesale_customers.standardize(wc_csv)
# wholesale_customers.scatter_plots(wc_csv)