#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten


class Cluster:

    def __init__(self):
        return

    THRESHOLD_CHANGE_DIST_TO_CENTROIDS = 0.1

    #
    # Get's optimal clusters, based on changes on sum of mean-square distance to centroids.
    # TODO: Need to include other factors like average centroid distance, average distance point-centroids, etc.
    # TODO: Otherwise this simple method below works but quite badly.
    #
    @staticmethod
    def get_optimal_cluster(matx, n_tries, iterations=50, threshold_change=THRESHOLD_CHANGE_DIST_TO_CENTROIDS, verbose=0):

        prev_sum_dist_to_centroids = 99999999.0
        prev_ratio_n_points_n_clusters = 99999999.0
        optimal_clusters = 1
        #
        # We try different cluster counts starting from 1 cluster
        #
        for i in range(1, n_tries+1, 1):
            # Cluster points in <i> clusters
            i_cluster_kmeans = kmeans(matx, k_or_guess=i, iter=iterations)
            i_cluster_idx = vq(obs=matx, code_book=i_cluster_kmeans[0])

            ratio_n_points_n_clusters = matx.shape[0] / i
            change_ratio_n_points_n_clusters = ((ratio_n_points_n_clusters - prev_ratio_n_points_n_clusters)
                                                / prev_ratio_n_points_n_clusters)

            #
            # Get sum of mean squares of distances of all points to their respective centroids
            #
            i_sum_dist_to_centroids = 0
            n = 0
            # Loop through all points
            for j in range(0, matx.shape[0], 1):
                j_point_vec = matx[j]
                j_point_cluster_no = i_cluster_idx[0][j]
                j_point_centroid = np.matrix(i_cluster_kmeans[0][j_point_cluster_no])
                tmp = j_point_vec - j_point_centroid
                j_distance_to_centroid = tmp*tmp.transpose()
                j_distance_to_centroid = j_distance_to_centroid[0,0] ** 0.5
                i_sum_dist_to_centroids = i_sum_dist_to_centroids + j_distance_to_centroid
                n = n + 1

            # Change (reduction) of sum of mean square of distances
            i_sum_dist_to_centroids = i_sum_dist_to_centroids
            change_sum_mean_square = ((i_sum_dist_to_centroids - prev_sum_dist_to_centroids) / prev_sum_dist_to_centroids)
            avg_dist_to_centroids = i_sum_dist_to_centroids / n
            optimal_clusters = i

            #
            # Calculate ratio of closest distance between centroids to furthest distance between point-centroid
            #
            closest_dist_btw_centroids = 99999999
            avg_dist_btw_centroids = 0
            n = 0
            for j in range(0, i, 1):
                for jj in range(j+1, i, 1):
                    dist_btw_centroid = i_cluster_kmeans[0][j] - i_cluster_kmeans[0][jj]
                    dist_btw_centroid = sum(np.multiply(dist_btw_centroid, dist_btw_centroid)) ** 0.5
                    if dist_btw_centroid < closest_dist_btw_centroids:
                        closest_dist_btw_centroids = dist_btw_centroid
                    avg_dist_btw_centroids = avg_dist_btw_centroids + dist_btw_centroid
                    n = n + 1
            if n>0:
                avg_dist_btw_centroids = avg_dist_btw_centroids / n
            if verbose>=1:
                print('Clusters ' + str(i) + ', change = ' + str(round(change_sum_mean_square*100,2)) + '%.' +
                      #'Closest distance between centroids = ' + str(round(closest_dist_btw_centroids, 2)) + '.' +
                      ' Avg dist btw centroids = ' + str(round(avg_dist_btw_centroids, 2)) + '.' +
                      ' Avg dist to centroids = ' + str(round(avg_dist_to_centroids, 2)) + '. ' +
                      ' Point to cluster ratio = ' + str(round(ratio_n_points_n_clusters, 2)) +
                      ' (' + str(round(change_ratio_n_points_n_clusters*100, 2)) + '% change).')
            if change_sum_mean_square > -threshold_change:
                break
            prev_sum_dist_to_centroids = i_sum_dist_to_centroids
            prev_ratio_n_points_n_clusters = ratio_n_points_n_clusters

        return optimal_clusters

    #
    # The main clustering function
    #
    @staticmethod
    def cluster(matx,
                feature_names,
                ncenters,
                iterations=50,
                verbose=0):
        # Set starting centers to be the top keywords
        ncols_matx = matx.shape[1]
        centers_initial = np.zeros((ncenters, ncols_matx))
        #for i in range(0, ncenters, 1):
        #    centers_initial[i][i] = 1
        cluster_kmeans = kmeans(matx, k_or_guess=ncenters, iter=iterations)
        cluster_idx = vq(obs=matx, code_book=cluster_kmeans[0])

        #
        # Keep cluster matrix in data frame
        #
        nrows = cluster_kmeans[0].shape[0]
        df_cluster_matrix = pd.DataFrame(data=cluster_kmeans[0], columns=feature_names, index=list(range(0, nrows, 1)))
        #print(df_cluster_matrix)

        point_cluster_no = [0]*matx.shape[0]
        point_distance_to_center = [0]*matx.shape[0]
        # Assign cluster number to lines
        if verbose >= 1:
            print('Assigning clusters to ' + str(matx.shape[0]) + ' sentences')
        for i in range(0, matx.shape[0], 1):
            point_vec = matx[i]
            cluster_no = cluster_idx[0][i]
            point_cluster_no[i] = cluster_no

            point_centroid = np.matrix(cluster_kmeans[0][cluster_no])
            tmp = point_vec - point_centroid
            distance_to_center = tmp*tmp.transpose()
            point_distance_to_center[i] = distance_to_center[0,0]**0.5

        # The cluster number of each point, in the same order as the original matrix
        df_point_cluster = pd.DataFrame({
                                        'ClusterNo': point_cluster_no,
                                        'DistanceToCenter': point_distance_to_center
                                        })
        return { 'ClusterMatrix': df_cluster_matrix, 'PointClusterInfo': df_point_cluster }


def demo_1():
    fn = ['a', 'b', 'c', 'd', 'e']
    m = np.matrix(data=np.zeros((8, 5)))
    m[0] = [1,2,1,0,0]
    m[1] = [2,1,2,0,0]
    m[2] = [1,1,1,0,0]
    m[3] = [1,0,0,1,1]
    m[4] = [2,0,0,1,2]
    m[5] = [0,10,10,0,10]
    m[6] = [0,9,11,0,12]
    m[7] = [0,10,9,0,10]

    # Optimal clusters
    optimal_clusters = Cluster.get_optimal_cluster(
        matx=m,
        n_tries=m.shape[0],
        verbose=1)
    print('Optimal Clusters = ' + str(optimal_clusters))

    retval = Cluster.cluster(matx=m,
                             feature_names=fn,
                             ncenters=3,
                             iterations=20,
                             verbose=1)
    print(retval['ClusterMatrix'])
    print(retval['PointClusterInfo'])
    return


#demo_1()
