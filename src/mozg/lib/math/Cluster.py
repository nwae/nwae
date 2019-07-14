#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten
import mozg.common.util.Log as lg
from inspect import currentframe, getframeinfo


class Cluster:

    def __init__(self):
        return

    THRESHOLD_CHANGE_DIST_TO_CENTROIDS = 0.1

    COL_CLUSTER_MATRIX = 'ClusterMatrix'
    COL_CLUSTER_NDARRY = 'ClusterNdarray'
    COL_CODE_BOOK      = 'PointClusterInfo'

    #
    # Get's optimal clusters, based on changes on sum of mean-square distance to centroids.
    # TODO: Need to include other factors like average centroid distance, average distance point-centroids, etc.
    # TODO: Otherwise this simple method below works but quite badly.
    #
    @staticmethod
    def get_optimal_cluster(
            matx,
            n_tries,
            iterations       = 50,
            threshold_change = THRESHOLD_CHANGE_DIST_TO_CENTROIDS
    ):
        lg.Log.debug(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Calculating optimal cluster for data\n\r' + str(matx) + '.'
        )
        prev_sum_dist_to_centroids = 99999999.0
        points_per_cluster_prev = 99999999.0
        optimal_clusters = 1
        #
        # We try different cluster counts starting from 1 cluster
        #
        for i in range(1, n_tries+1, 1):
            lg.Log.debugdebug(
                str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Trying ' + str(i) + ' cluster(s)...'
            )
            # Cluster points in <i> clusters
            i_cluster_kmeans = kmeans(matx, k_or_guess=i, iter=iterations)
            # Code book when assigning a cluster back to the samples
            i_cluster_idx = vq(obs=matx, code_book=i_cluster_kmeans[0])
            lg.Log.debug(
                str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r: Kmeans \n\r' + str(i_cluster_kmeans)
                + '\n\r, Indexes \n\r' + str(i_cluster_idx) + '.'
            )

            #
            # How many points per cluster on average
            #
            points_per_cluster = matx.shape[0] / i
            rate_of_change_points_per_cluster =\
                ((points_per_cluster - points_per_cluster_prev) / points_per_cluster_prev)

            #
            # Get sum of mean squares of distances of all points to their respective centroids
            #
            i_sum_dist_to_centroids = 0
            n = 0
            # Loop through all points
            for j in range(0, matx.shape[0], 1):
                j_point_vec = matx[j]
                j_point_cluster_no = i_cluster_idx[0][j]
                j_point_centroid = np.array(i_cluster_kmeans[0][j_point_cluster_no], ndmin=2)
                # If this is a column matrix, with >1 rows, and a single column, transpose to become a row matrix
                if (j_point_centroid.shape[0] > 1) and (j_point_centroid.shape[1] == 1):
                    j_point_centroid = j_point_centroid.transpose()
                tmp = j_point_vec - j_point_centroid
                j_distance_to_centroid = tmp*tmp.transpose()
                j_distance_to_centroid = j_distance_to_centroid[0,0] ** 0.5
                i_sum_dist_to_centroids = i_sum_dist_to_centroids + j_distance_to_centroid
                n = n + 1
                lg.Log.debugdebug(
                    str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Sample/Point ' + str(np.round(j_point_vec,3)) + ' in cluster ' + str(j_point_cluster_no)
                    + ', centroid ' + str(np.round(j_point_centroid,3))
                    + ', Distance = ' + str(round(j_distance_to_centroid,3))
                    + ', sum distance = ' + str(round(i_sum_dist_to_centroids,3)) + '.'
                )

            lg.Log.debugdebug(
                str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': ' + str(i) + ' clusters. Sum distance to centroid = ' + str(i_sum_dist_to_centroids) + '.'
            )
            # Change (reduction) of sum of mean square of distances
            reduction_pct_sum_mean_square =\
                - ((i_sum_dist_to_centroids - prev_sum_dist_to_centroids) / prev_sum_dist_to_centroids)
            avg_dist_to_centroids = i_sum_dist_to_centroids / n
            optimal_clusters = i

            #
            # Calculate ratio of closest distance between centroids to furthest distance between point-centroid
            #
            # Ideally "far"
            closest_dist_btw_centroids = 99999999
            # Ideally "well separated"
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
            lg.Log.debugdebug(
                str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + '\n\r' + str(i) + ' Clusters'
                + '\n\r\tReduction % of sum mean square (from ' + str(i-1) + ' clusters) = '
                + str(round(reduction_pct_sum_mean_square*100,2)) + '%.'
                + '\n\r\tAvg dist btw centroids = ' + str(round(avg_dist_btw_centroids, 2)) + '.'
                + '\n\r\tClosest dist btw centroids = ' + str(round(closest_dist_btw_centroids, 2)) + '.'
                + '\n\r\tAvg dist to centroids = ' + str(round(avg_dist_to_centroids, 2)) + '. '
                + '\n\r\tPoints per cluster = ' + str(round(points_per_cluster, 2))
                + ' (' + str(round(rate_of_change_points_per_cluster*100, 2)) + '% change).')

            #
            # Simple Criteria for Optimal Cluster Count
            # - If the change of the sum mean square is becoming too small, means we are already optimal
            # TODO Add criteria such that average distance between centroids don't become too "close"
            # TODO Add criteria such that closest distance between centroids is "much bigger" than avg distance to centroids
            #
            if reduction_pct_sum_mean_square < threshold_change:
                break
            prev_sum_dist_to_centroids = i_sum_dist_to_centroids
            points_per_cluster_prev = points_per_cluster

        return optimal_clusters

    #
    # The main clustering function
    #
    @staticmethod
    def cluster(
            matx,
            feature_names,
            ncenters,
            iterations=50
    ):
        lg.Log.debug(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start clustering ncenters=' + str(ncenters) + ', data=\n\r' + str(matx)
        )
        # Set starting centers to be the top keywords
        ncols_matx = matx.shape[1]
        # centers_initial = np.zeros((ncenters, ncols_matx))
        cluster_kmeans = kmeans(matx, k_or_guess=ncenters, iter=iterations)
        cluster_idx = vq(obs=matx, code_book=cluster_kmeans[0])

        #
        # Keep cluster matrix in data frame
        #
        nrows = cluster_kmeans[0].shape[0]
        np_cluster = np.array(cluster_kmeans[0])
        df_cluster_matrix = pd.DataFrame(data=cluster_kmeans[0], columns=feature_names, index=list(range(0, nrows, 1)))
        lg.Log.debugdebug(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Cluster matrix\n\r' + str(df_cluster_matrix)
        )

        point_cluster_no = [0]*matx.shape[0]
        point_distance_to_center = [0]*matx.shape[0]
        # Assign cluster number to lines
        lg.Log.info(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Assigning clusters to ' + str(matx.shape[0]) + ' sentences...'
        )
        for i in range(0, matx.shape[0], 1):
            point_vec = np.array(matx[i], ndmin=1)
            lg.Log.debugdebug('point vec: ' + str(point_vec))
            cluster_no = cluster_idx[0][i]
            point_cluster_no[i] = cluster_no

            point_centroid = np.array(cluster_kmeans[0][cluster_no], ndmin=1)
            lg.Log.debugdebug('point centroid: ' + str(point_centroid))

            tmp = point_vec - point_centroid
            lg.Log.debugdebug('Dif = ' + str(tmp))
            distance_to_center = np.sum(np.multiply(tmp,tmp))
            point_distance_to_center[i] = distance_to_center**0.5

        # The cluster number of each point, in the same order as the original matrix
        df_point_cluster = pd.DataFrame({
            'ClusterNo': point_cluster_no,
            'DistanceToCenter': point_distance_to_center
        })
        lg.Log.info(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Final Cluster\n\r ' + str(df_point_cluster)
        )
        return {
            Cluster.COL_CLUSTER_NDARRY: np_cluster,
            Cluster.COL_CLUSTER_MATRIX: df_cluster_matrix,
            Cluster.COL_CODE_BOOK:      df_point_cluster
        }


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_2
    fn = ['a', 'b', 'c', 'd', 'e']
    m = np.zeros((8, 5))
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
        n_tries=m.shape[0]
    )
    print('Optimal Clusters = ' + str(optimal_clusters))

    retval = Cluster.cluster(
        matx=m,
        feature_names=fn,
        ncenters=3,
        iterations=20
    )
    print(retval[Cluster.COL_CLUSTER_NDARRY])
    print(retval[Cluster.COL_CLUSTER_MATRIX])
    print(retval[Cluster.COL_CODE_BOOK])
    exit(0)
