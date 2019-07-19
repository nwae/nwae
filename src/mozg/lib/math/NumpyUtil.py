import numpy as np
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo


class NumpyUtil:

    def __init__(self):
        return

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
