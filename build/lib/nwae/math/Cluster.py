#!/usr/bin/python
# -*- coding: utf-8 -*-

# !!! Will work only on Python 3 and above

import numpy as np
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten
import nwae.utils.Log as lg
from inspect import currentframe, getframeinfo
import nwae.utils.UnitTest as ut

#
# Some concepts:
#
# 1. Cluster Radius
#    For all classes, or cluster the radius of the class/cluster is defined as the distance of the
#    "center" of the class/cluster to the furthest point.
#
class Cluster:

    def __init__(self):
        return

    THRESHOLD_CHANGE_DIST_TO_CENTROIDS = 0.1

    COL_CLUSTER_LABEL = 'ClusterLabel'
    COL_CLUSTER_RADIUS = 'ClusterRadius'

    #
    # The return object for our cluster function
    #
    class ReturnCluster:
        def __init__(
                self,
                np_cluster_centers,
                np_cluster_labels,
                df_distances_point_to_cc,
                np_cluster_radius,
                val_max_cluster_radius
        ):
            self.np_cluster_centers = np_cluster_centers
            self.np_cluster_labels = np_cluster_labels,
            self.df_distances_point_to_cc = df_distances_point_to_cc
            self.np_cluster_radius = np_cluster_radius
            self.val_max_cluster_radius = val_max_cluster_radius
            return

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
            ncenters,
            feature_names = None,
            iterations    = 10
    ):
        lg.Log.debugdebug(
            str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Start clustering ncenters=' + str(ncenters) + ', data=\n\r' + str(matx)
            + ', shape ' + str(matx.shape) + '.'
        )
        if ncenters > matx.shape[0]:
            ncenters = matx.shape[0]

        if feature_names is not None:
            if len(feature_names) != matx.shape[1]:
                raise Exception(
                    str(Cluster.__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Number of columns for matx of shape ' + str(matx.shape)
                    + ' not equal to feature name columns ' + str(len(feature_names)) + '.'
                )

        # Set starting centers to be the top keywords
        # ncols_matx = matx.shape[1]
        # centers_initial = np.zeros((ncenters, ncols_matx))
        cluster_kmeans = kmeans(matx, k_or_guess=ncenters, iter=iterations)
        cluster_idx = vq(obs=matx, code_book=cluster_kmeans[0])
        lg.Log.debugdebug(
            'Cluster kmeans:\n\r' + str(cluster_kmeans)
            + '\n\r, and cluster vq:\n\r' + str(cluster_idx)
        )

        np_cluster_centers= np.array(cluster_kmeans[0])
        np_cluster_labels = np.array(cluster_idx[0])
        np_unique_cluster_labels = np.array(range(np_cluster_centers.shape[0]))

        # Each row represents the points (indexes only), and columns the distance to the respective column center
        df_dist_to_cluster_centers = pd.DataFrame(
            data = np.zeros(shape=(matx.shape[0], np_unique_cluster_labels.shape[0])),
            columns = np_unique_cluster_labels,
            index   = range(matx.shape[0])
        )
        df_dist_to_cluster_centers = df_dist_to_cluster_centers.transpose()

        # Get distance of all points to all cluster centers
        for idx in np_unique_cluster_labels:
            cc = np_cluster_centers[idx]
            #print('cc: ' + str(cc))
            np_cc = np.array([cc]*matx.shape[0])
            #print('np_cc:\n\r' + str(np_cc))
            dist = matx - np_cc
            # Square every array element
            dist = np.power(dist, 2)
            # Sum every row to create a single column matrix
            dist = dist.sum(axis=1)
            # Take the square root of every element in the single column matrix as distance
            dist = np.power(dist, 0.5)
            # Convert to a row matrix
            dist = dist.transpose()
            #print('dist:' + str(dist))
            df_dist_to_cluster_centers.at[idx] = dist

        # Transpose back
        df_dist_to_cluster_centers = df_dist_to_cluster_centers.transpose()
        lg.Log.debugdebug(
            'Cluster Code Book:\n\r' + str(np_cluster_labels)
            + '\n\rDistance to Cluster Centers:\n\r' + str(df_dist_to_cluster_centers)
        )

        #
        # Get cluster radius &
        #
        df_cluster_radius = pd.DataFrame(
            data = {
                Cluster.COL_CLUSTER_LABEL: np_unique_cluster_labels,
                Cluster.COL_CLUSTER_RADIUS: [0.0] * np_unique_cluster_labels.shape[0]
            }
        )
        for idx in np_unique_cluster_labels:
            # Get just this cluster points distances
            dist_cluster = df_dist_to_cluster_centers[np_cluster_labels==idx]
            # Just the cluster column
            cluster_col = np.array(dist_cluster[idx])
            # Max of the label column
            max_dist = max(cluster_col)
            df_cluster_radius[Cluster.COL_CLUSTER_RADIUS].at[idx] = max_dist

        lg.Log.debugdebug(
            'Cluster Radius:\n\r' + str(df_cluster_radius)
        )

        retobj = Cluster.ReturnCluster(
            np_cluster_centers = np_cluster_centers,
            np_cluster_labels  = np_cluster_labels,
            df_distances_point_to_cc = df_dist_to_cluster_centers,
            np_cluster_radius = np.array(df_cluster_radius[Cluster.COL_CLUSTER_RADIUS]),
            val_max_cluster_radius = max(df_cluster_radius[Cluster.COL_CLUSTER_RADIUS])
        )

        return retobj


class ClusterUnitTest:
    def __init__(
            self,
            ut_params
    ):
        self.ut_params = ut_params
        if self.ut_params is None:
            # We only do this for convenience, so that we have access to the Class methods in UI
            self.ut_params = ut.UnitTestParams()
        return

    def __extract_cluster_groups_from_points_and_labels(
            self,
            # np array of points, e.g. [[0,1], [0,1.1], [2,1], [2,0], [2,0.4]]
            point_array,
            # Their group labels, e.g. [0,0,0,1,1]
            point_group_labels
    ):
        lg.Log.debug('Input point array: ' + str(point_array))
        lg.Log.debug('Input group labels: ' + str(point_group_labels))
        # Extract out the same group label points into separate lists
        cluster_groups_dict = {}
        track_group_labels = []
        for i in range(len(point_array)):
            group_label = point_group_labels[i]
            # We only add a new group when we see a new label, this way we keep the order
            # of the groups the same, regardless of the different labels like (0,0,1,1,1)
            # or (5,5,7,7,7) - the extraction of the groups remain the same
            if group_label not in track_group_labels:
                track_group_labels.append(group_label)
                cluster_groups_dict[group_label] = []
            # Add the point index i in the cluster group label
            cluster_groups_dict[group_label].append(i)

        cluster_centers = []
        count = 0
        for g in cluster_groups_dict.values():
            group_size = len(g)
            cluster_centers.append(
                np.sum(point_array[count:(count + group_size)], axis=0) / group_size
            )
            count += group_size

        cluster_groups = list(cluster_groups_dict.values())
        lg.Log.info(
            '\n\rCluster groups: ' + str(cluster_groups)
            + '\n\rCluster centers: ' + str(cluster_centers)
        )
        return cluster_groups, cluster_centers

    def __test_data(
            self,
            # np array of points, e.g. [[0,1], [0,1.1], [2,1], [2,0], [2,0.4]]
            point_array,
            # Their group labels, e.g. [0,0,0,1,1]
            point_group_labels
    ):
        res_test = ut.ResultObj(count_ok=0, count_fail=0)

        expected_groups, expected_centers = self.__extract_cluster_groups_from_points_and_labels(
            point_array = point_array,
            point_group_labels = point_group_labels
        )

        # Optimal clusters
        optimal_clusters = Cluster.get_optimal_cluster(
            matx    = point_array,
            n_tries = point_array.shape[0]
        )
        lg.Log.debug('Optimal Clusters = ' + str(optimal_clusters))

        retval = Cluster.cluster(
            matx       = point_array,
            ncenters   = len(expected_centers),
            iterations = 20
        )

        lg.Log.debug('Cluster Centers:\n\r' + str(retval.np_cluster_centers))
        lg.Log.debug('Cluster Labels:\n\r' + str(retval.np_cluster_labels))
        lg.Log.debug('Cluster Point Distances to CC:\n\r' + str(retval.df_distances_point_to_cc))
        lg.Log.debug('Cluster Radius:\n\r' + str(retval.np_cluster_radius))
        lg.Log.debug('Max Cluster Radius:\n\r' + str(retval.val_max_cluster_radius))

        #
        # Compare cluster centers with expected cluster centers
        #
        observed_groups, observed_centers = self.__extract_cluster_groups_from_points_and_labels(
            point_array = point_array,
            point_group_labels = retval.np_cluster_labels[0]
        )

        #
        # Compare groups is easy
        #
        res_test.update_bool(res_bool=ut.UnitTest.assert_true(
            observed = observed_groups,
            expected = expected_groups,
            test_comment = 'Compare observed: ' + str(observed_groups)
                           + ', and expected groups: ' + str(expected_groups)
        ))

        #
        # Comparing centers need to do one by one with error tolerance
        #
        for i in range(len(expected_centers)):
            exp_ctr = expected_centers[i]
            obs_ctr = observed_centers[i]
            euclidean_dist = np.sum( (exp_ctr - obs_ctr)**2 ) ** 0.5
            res_test.update_bool(res_bool=ut.UnitTest.assert_true(
                observed = euclidean_dist < 0.00001,
                expected = True,
                test_comment = 'Center ' + str(i)
                               + '. Compare distance = ' + str(euclidean_dist)
                               + ' of observed center ' + str(obs_ctr)
                               + ', and expected center: ' + str(exp_ctr)
            ))

        return res_test

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        m = np.zeros((9, 5))
        m[0] = [1, 2, 1, 0, 0]
        m[1] = [2, 1, 2, 0, 0]
        m[2] = [1, 1, 1, 0, 0]
        m[3] = [1, 0, 0, 1, 1]
        m[4] = [2, 0, 0, 1, 2]
        m[5] = [0, 10, 10, 0, 10]
        m[6] = [0, 9, 11, 0, 12]
        m[7] = [0, 10, 9, 0, 10]
        m[8] = [0, 10, 9, 0, 10]
        res = self.__test_data(
            point_array = m,
            point_group_labels = (0, 0, 0, 1, 1, 2, 2, 2, 2)
        )
        res_final.update(other_res_obj=res)

        return res_final


if __name__ == '__main__':
    lg.Log.LOGLEVEL = lg.Log.LOG_LEVEL_DEBUG_1

    ClusterUnitTest(ut_params=None).run_unit_test()
    exit(0)
