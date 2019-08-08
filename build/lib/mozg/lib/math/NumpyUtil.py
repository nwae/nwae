import numpy as np
import mozg.utils.Log as log
from inspect import currentframe, getframeinfo
import mozg.utils.Profiling as prf


class NumpyUtil:

    def __init__(self):
        return

    @staticmethod
    def get_point_pixel_count(x):
        n_pixels = 1
        i = 1
        # Ignore 1st dimension (index=0) which is the point index
        while i < x.ndim:
            n_pixels *= x.shape[i]
            i += 1
        return n_pixels

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


if __name__ == '__main__':
    log.Log.LOGLEVEL = log.Log.LOG_LEVEL_DEBUG_1
    arr = np.array(range(10))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=1))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=2))
    print(NumpyUtil.convert_dimension(arr=arr, to_dim=3))

    arr3 = np.array([[[1,2,3,4,5],[9,8,7,6,5]]])
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=3))
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=2))
    print(NumpyUtil.convert_dimension(arr=arr3, to_dim=1))
