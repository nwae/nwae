import numpy as np
import nwae.utils.Log as log
from inspect import currentframe, getframeinfo
import nwae.utils.Profiling as prf
import nwae.math.Constants as const
import pandas as pd
import nwae.utils.UnitTest as ut


class NumpyUtil:

    def __init__(self):
        return

    #
    # Multiplies the dimensions (not including the first)
    #
    @staticmethod
    def get_point_pixel_count(x):
        n_pixels = 1
        i = 1
        # Ignore 1st dimension (index=0) which is the point index
        while i < x.ndim:
            n_pixels *= x.shape[i]
            i += 1
        return n_pixels

    #
    # Converts to desired numpy array dimension, usually to higher dimension.
    # Returns a new numpy ndarray object
    #
    @staticmethod
    def convert_dimension(
            arr,
            to_dim
    ):
        if to_dim < 1:
            to_dim = 1
        cur_dim = arr.ndim

        if cur_dim == to_dim:
            return np.array(arr)
        elif cur_dim > to_dim:
            # Reduce dimension
            while arr.ndim > to_dim:
                cur_dim = arr.ndim
                # Join the rows
                arr_new = None
                for i in range(arr.shape[0]):
                    log.Log.debugdebug('****** Append\n\r' + str(arr[i]))
                    if arr_new is None:
                        arr_new = np.array(arr[i], ndmin=cur_dim-1)
                    else:
                        arr_new = np.append(arr_new, arr[i])
                arr = arr_new
                log.Log.debugdebug('*** Loop:\n\r' + str(arr))
        else:
            # Increase dimension
            while arr.ndim < to_dim:
                arr = np.array([arr])

        return arr

    #
    # For points on a hypersphere, 1-dotproduct is the angle distance
    # The point v and points in the array x are assumed to be already on the positive hypersphere.
    # If not this metric has no meaning.
    #
    # In neural network interpretation this is equivalent to an input of dimension v,
    # and a layer with x.shape[0] nodes.
    # Each node then has weights x[node] which is the plane perpendicular to the reference
    # point on the hypersphere.
    #
    @staticmethod
    def calc_metric_angle_distance(
            # Point
            v,
            # Array of points
            x,
            do_profiling = False
    ):
        prf_start = None
        if do_profiling:
            prf_start = prf.Profiling.start()

        if (type(v) is not np.ndarray) or (type(x) is not np.ndarray):
            errmsg = str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno)\
                     + ': Excepted numpy ndarray type, got type v as "' + str(type(v))\
                     + ', and type x_ref as "' + str(type(x))
            log.Log.error(errmsg)
            raise Exception(errmsg)

        element_product = np.multiply(x, v)
        cosine_angle = np.sum(element_product, axis=1)
        # log.Log.debugdebug(
        #     str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno) \
        #     + ': Dot product between x:\n\r' + str(x)
        #     + '\n\rand v:\n\r' + str(v)
        #     + '\n\relement product:\n\r' + str(element_product)
        #     + '\n\rcosine angle:\n\r' + str(cosine_angle)
        # )
        if do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            log.Log.important(
                str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING calc_metric_inverse_dot_product(): ' + str(round(1000*prf_dur,0))
                + ' milliseconds.'
            )

        #
        # We assume all x are positive and v also, thus cosine angle should be between [0,1]
        #
        bad_rows = (cosine_angle<0.0) | (cosine_angle>1.0)
        sum_bad_rows = np.sum(1*bad_rows)
        if sum_bad_rows > 0:
            log.Log.warning(
                str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno)
                + ': Bad Rows produced from dot product between x:\n\r' + str(x[bad_rows].tolist())
                + '\n\rand v:\n\r' + str(v.tolist())
                + '\n\relement product:\n\r' + str(element_product[bad_rows])
                + '\n\rcosine angle:\n\r' + str(cosine_angle[bad_rows])
            )
            cosine_angle[cosine_angle<0.0] = 0.0
            cosine_angle[cosine_angle>1.0] = 1.0

        #
        # In neural network form, we would be returning instead 1-cosine_angle, which
        # is the perpendicular distance from the plane.
        #
        angle_distance = np.arccos(cosine_angle)
        log.Log.debugdebug(
            str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno) \
            + ': Dot product between x:\n\r' + str(x)
            + '\n\rand v:\n\r' + str(v)
            + '\n\relement product:\n\r' + str(element_product)
            + '\n\rcosine angle:\n\r' + str(cosine_angle)
            + '\n\rangle:\n\r' + str(angle_distance)
        )

        return angle_distance


    #
    # Calculates the normalized distance (0 to 1 magnitude range) of a point v (n dimension)
    # to a set of references (n+1 dimensions or k rows of n dimensional points) by knowing
    # the theoretical max/min of our hypersphere
    #
    @staticmethod
    def calc_distance_of_point_to_x_ref(
            # Point
            v,
            x_ref,
            y_ref,
            do_profiling = False
    ):
        prf_start = None
        if do_profiling:
            prf_start = prf.Profiling.start()

        if (type(v) is not np.ndarray) or (type(x_ref) is not np.ndarray):
            errmsg = str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno)\
                     + ': Excepted numpy ndarray type, got type v as "' + str(type(v))\
                     + ', and type x_ref as "' + str(type(x_ref))
            log.Log.error(errmsg)
            raise Exception(errmsg)

        log.Log.debugdebug('Evaluate distance between v: ' + str(v) + ' and\n\r' + str(x_ref))

        #
        # Remove rows of x_ref with no common features
        # This can almost half the time needed for calculation
        #
        relevant_columns = v>0
        relevant_columns = NumpyUtil.convert_dimension(arr=relevant_columns, to_dim=1)
        # Relevant columns of x_ref extracted
        log.Log.debugdebug('Relevant columns:\n\r' + str(relevant_columns))
        x_ref_relcols = x_ref.transpose()[relevant_columns].transpose()
        # Relevant rows, those with sum of row > 0
        x_ref_relrows = np.sum(x_ref_relcols, axis=1) > 0
        x_ref_rel = x_ref[x_ref_relrows]
        y_ref_rel = y_ref[x_ref_relrows]

        v_ok = NumpyUtil.convert_dimension(arr=v, to_dim=2)
        # if v.ndim == 1:
        #     # Convert to 2 dimensions
        #     v_ok = np.array([v])

        # Create an array with the same number of rows with rfv
        vv = np.repeat(a=v_ok, repeats=x_ref_rel.shape[0], axis=0)
        log.Log.debugdebug('vv repeat: ' + str(vv))

        dif = vv - x_ref_rel
        log.Log.debugdebug('dif with x_ref: ' + str(dif))

        # Square every element in the matrix
        dif2 = np.power(dif, 2)
        log.Log.debugdebug('dif squared: ' + str(dif2))

        # Sum every row to create a single column matrix
        dif2_sum = dif2.sum(axis=1)
        log.Log.debugdebug('dif aggregated sum: ' + str(dif2_sum))

        # Take the square root of every element in the single column matrix as distance
        distance_x_ref = np.power(dif2_sum, 0.5)
        log.Log.debugdebug('distance to x_ref: ' + str(distance_x_ref))

        # Convert to a single row matrix
        distance_x_ref = distance_x_ref.transpose()
        log.Log.debugdebug('distance transposed: ' + str(distance_x_ref))

        if do_profiling:
            prf_dur = prf.Profiling.get_time_dif(prf_start, prf.Profiling.stop())
            log.Log.important(
                str(NumpyUtil.__name__) + str(getframeinfo(currentframe()).lineno)
                + ' PROFILING calc_distance_of_point_to_x_ref(): ' + str(round(1000*prf_dur,0))
                + ' milliseconds.'
            )

        class retclass:
            def __init__(self, distance_x_rel, y_rel):
                self.distance_x_rel = distance_x_rel
                self.y_rel = y_rel

        return retclass(
            distance_x_rel = distance_x_ref,
            y_rel          = y_ref_rel
        )

    @staticmethod
    def normalize(
            x,
            axis  = -1,
            # Default Frobenius norm
            order = 2
    ):
        norms = np.linalg.norm(
            x    = x,
            # If 2D matrix, use axis=1 to calculate norm of the rows.
            # Thus in the 2D matrix case it loops by fixed axis=0, thus accessing the rows
            # If n-dimensional, use axis=n-1 to calculate the norm of the last 1D rows,
            # as it fixes and loops by the first n-1 axis to get norm
            axis = axis,
            ord  = order
        )
        log.Log.debugdebug(
            'Norms:\n\r' + str(norms)
        )
        l2 = np.atleast_1d(norms)
        log.Log.debugdebug(
            'Norms (at least 1D):\n\r' + str(norms)
        )
        # Make 0 norms become 1
        l2[l2 == 0] = 1
        log.Log.debugdebug(
            'Norms (replace 0s with 1s):\n\r' + str(norms)
        )

        #
        # Finally do a division but with same d
        #
        xn = x / np.expand_dims(l2, axis)
        log.Log.debugdebug(
            'Normalized vectors:\n\r' + str(xn)
        )
        return xn

    @staticmethod
    def is_normalized(
            x
    ):
        return (abs((np.sum(np.multiply(x, x)) ** 0.5) - 1) < const.Constants.SMALL_VALUE)

    #
    # Returns biggest to smallest indexes, given a numpy ndarray of 1 dimension
    #
    @staticmethod
    def get_top_indexes(
            # numpy ndarray of a single dimension only
            data,
            # By default, from biggest to smallest
            ascending = False,
            top_x = 5
    ):
        if data.ndim > 1:
            raise Exception(
                'Expected single dimension, got ' + str(data.ndim) + ' dimensional: '
                + str(data)
            )
        len_data = data.shape[0]
        if len_data == 0:
            raise Exception(
                'Empty data'
            )
        df = pd.DataFrame(data=data, columns=['data'])
        df.sort_values(by=['data'], ascending=ascending, inplace=True)
        # print(df)
        top_indexes = np.array(df.index[0:min(top_x, len_data)])
        # print(top_x)
        return top_indexes


class NumpyUtilUnittest:
    def __init__(self, ut_params):
        self.ut_params = ut_params
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        x_input = np.array([
            [5.6, 2.4, 55.6, 6.8],
            [55, 66, 22, 77, 33, 22],
        ], dtype=object)
        out_expected = np.array([
            [2, 3, 0, 1],
            [3, 1, 0, 4, 2],
        ], dtype=object)
        for i in range(x_input.shape[0]):
            data = np.array(x_input[i])
            expected = np.array(out_expected[i])
            top_indexes = NumpyUtil.get_top_indexes(
                data = data,
                ascending = False,
                top_x = 5
            )
            res_final.update_bool(ut.UnitTest.assert_true(
                observed = list(top_indexes),
                expected = list(expected),
                test_comment = 'test ' + str(data)
            ))

        log.Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Numpy Util Unit Test PASSED ' + str(res_final.count_ok) + ', FAILED ' + str(res_final.count_fail)
        )
        return res_final


if __name__ == '__main__':
    res = NumpyUtilUnittest(ut_params=None).run_unit_test()
    exit(0)

    A = np.random.randn(3, 3, 3)
    print('Test Normalization of: ' + str(A))
    print('Result: ')
    print(NumpyUtil.normalize(x=A, axis=0))
    print(NumpyUtil.normalize(x=A, axis=1))
    print(NumpyUtil.normalize(x=A, axis=2))

    print(NumpyUtil.normalize(np.arange(3)[:, None]))
    print(NumpyUtil.normalize(np.arange(3)))

    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_2
    arr = np.array(range(10))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=1))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=2))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=3))

    arr3 = np.array([[[1,2,3,4,5],[9,8,7,6,5]]])
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=3))
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=2))
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=1))

    print(NumpyUtil.normalize(
        x = np.array([
            [1,2,3,4,5],
            [5,3,2,6,7],
            [1,1,1,1,1],
            [0,0,0,0,0]
        ]),
        axis = 1
    ))
