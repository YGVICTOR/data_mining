import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from collections import defaultdict
from matplotlib import pyplot as plt

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    data_df = pd.read_csv(data_file)
    columns = list(data_df.columns)
    if 'Channel' in columns:
        data_df.drop(['Channel'], axis=1, inplace=True)
    if 'Region' in columns:
        data_df.drop(['Region'], axis=1, inplace=True)
    return data_df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    mean_series = df.mean(axis=0)
    std_series = df.std(axis=0)
    min_series = df.min(axis=0)
    max_series = df.max(axis=0)
    data = {'mean': mean_series, 'std': std_series, 'min': min_series, 'max': max_series}
    summary_df = pd.DataFrame(data)
    summary_df['mean'] = summary_df['mean'].apply(lambda x: round(x))
    summary_df['std'] = summary_df['std'].apply(lambda x: round(x))
    return summary_df


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    df_copy = df.copy(deep=True)
    return (df_copy-df_copy.mean())/df_copy.std()


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    # initialise the KMeans clustering object
    km = cluster.KMeans(init='random', n_init=1, n_clusters=k, max_iter=30, tol=0.0001)
    # compute the clusters:
    km.fit(df)
    return pd.Series(km.labels_)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    # initialise the KMeans clustering object
    km = cluster.KMeans( n_init=1, n_clusters=k, max_iter=30, tol=0.0001)
    # compute the clusters:
    km.fit(df)
    return pd.Series(km.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    ac = cluster.AgglomerativeClustering(n_clusters=k, linkage='average', affinity='euclidean')
    # compute the cluster
    ac.fit(df)
    return pd.Series(ac.labels_)


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X, y):
    sc = metrics.silhouette_score(X, y, metric='euclidean')
    return sc


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    # store the best model
    evaluation_dict = {'Algorithm': [], 'data': [], 'k': [], 'Silhouette Score': []}

    # Divide the data points into k clusters, for k ∈ {3, 5, 10}
    k_list = [3, 5, 10]
    # define repeat times
    repeat = 10
    # prepare standardize df
    standardize_df = standardize(df)
    # using kmeans and agglomerative hierarchical clustering
    for k in k_list:
        # agglomerative + Original
        agglomerative_labels = agglomerative(df, k)
        current_Silhouette = clustering_score(df, agglomerative_labels)
        # update evaluation_dict
        evaluation_dict['Algorithm'].append('Agglomerative')
        evaluation_dict['data'].append('Original')
        evaluation_dict['k'].append(k)
        evaluation_dict['Silhouette Score'].append(current_Silhouette)

        # agglomerative + Standardized
        agglomerative_after_normalization_labels = agglomerative(standardize_df, k)
        current_Silhouette = clustering_score(standardize_df, agglomerative_after_normalization_labels)
        evaluation_dict['Algorithm'].append('Agglomerative')
        evaluation_dict['data'].append('Standardized')
        evaluation_dict['k'].append(k)
        evaluation_dict['Silhouette Score'].append(current_Silhouette)

        # print all 10 repetitions
        for current_repeat in range(repeat):
            # k_means before normalization
            k_means_labels = kmeans(df, k)
            current_Silhouette = clustering_score(df, k_means_labels)
            # update evaluation_dict
            evaluation_dict['Algorithm'].append('Kmeans')
            evaluation_dict['data'].append('Original')
            evaluation_dict['k'].append(k)
            evaluation_dict['Silhouette Score'].append(current_Silhouette)
            # k_meas after normalizaiton
            k_means_after_normalization_labels = kmeans(standardize_df, k)
            current_Silhouette = clustering_score(standardize_df, k_means_after_normalization_labels)
            # update evaluation_dict
            evaluation_dict['Algorithm'].append('Kmeans')
            evaluation_dict['data'].append('Standardized')
            evaluation_dict['k'].append(k)
            evaluation_dict['Silhouette Score'].append(current_Silhouette)
    return pd.DataFrame(evaluation_dict)


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    rdf = cluster_evaluation(df)
    best_score = best_clustering_score(rdf)
    tmp_result = rdf[rdf['Silhouette Score'] == best_score].iloc[0]
    algorithm = tmp_result['Algorithm']
    data = tmp_result['data']
    k = tmp_result['k']
    if data == 'Original':
        data_source = df
    else:
        data_source = standardize(df)
    if algorithm == 'Agglomerative':
        predicted_labels = agglomerative(data_source, k)
    else:
        best_Silhouette = 0
        for current_repeat in range(10):
            # k_means before normalization
            k_means_labels = kmeans(data_source, k)
            current_Silhouette = clustering_score(data_source, k_means_labels)
            if current_Silhouette > best_Silhouette:
                best_Silhouette = current_Silhouette
                predicted_labels = k_means_labels
    # start to plot
    colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    labels = predicted_labels.unique()
    data_source['predicted_labels'] = predicted_labels

    # save 15 individual format
    for x_index in range(len(data_source.columns)-1):
        for y_index in range(x_index+1,len(data_source.columns)-1):
            x_label = data_source.columns[x_index]
            y_label = data_source.columns[y_index]
            fig = plt.figure()
            for i in range(len(labels)):
                plt.scatter(x=data_source[data_source['predicted_labels'] == labels[i]][x_label], y=data_source[data_source['predicted_labels'] == labels[i]][y_label],color=colors[i])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(labels)
            plt.title('{x_label}-{y_label}'.format(x_label=x_label,y_label=y_label))
            plt.savefig('{x_label}-{y_label}.png'.format(x_label=x_label,y_label=y_label))