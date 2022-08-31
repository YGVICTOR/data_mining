import wholesale_customers
if __name__ == '__main__':
    df = wholesale_customers.read_csv_2('../data/wholesale_customers.csv')
    print(df)
    print(wholesale_customers.summary_statistics(df))
    print(wholesale_customers.standardize(df))
    # print(df)
    # print(wholesale_customers.kmeans(df,3))
    Y = wholesale_customers.agglomerative(df,3)
    # print(wholesale_customers.clustering_score(df,Y))
    rdf = wholesale_customers.cluster_evaluation(df)
    print(rdf)
    print(wholesale_customers.best_clustering_score(rdf))
    # wholesale_customers.scatter_plots(df)
