import numpy as np
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import nwae.utils.UnitTest as ut


class CenterOfMass:

    def __init__(
            self,
            x
    ):
        self.x = x
        assert type(x) == np.ndarray
        return

    def calculate(
            self
    ):
        self.x_shape = self.x.shape

        # How many elements altogether
        self.x_elements_count = np.product(self.x_shape)
        self.x_dim = len(self.x_shape)

        if self.x_elements_count == 0:
            return np.array([np.nan]*self.x_dim)

        # No negative numbers
        assert np.min(self.x) >= 0
        assert self.x_dim > 0
        assert self.x_elements_count > 0

        # Keep the dimension coordinates here
        self.x_coordinates = np.zeros(shape=[self.x_dim] + [self.x_elements_count])

        # For example if x has shape (4,3,2), this number will start with 4*3*2 = 24
        repeat_times = self.x_elements_count
        for dim in range(self.x_dim):
            # For example first dimension will have a scalar repeat 3*2=6 times (0,0,0,0,0,0,1,1,1,1,1,1,..)
            # as each row will have 6 elements in total,
            # 2nd dimension will repeat 2 times (0,0,1,1,2,2,..) as each row will have 2 elements in total
            repeat_times = repeat_times / self.x_shape[dim]
            # Each number 0, 1, 2, ... is repeated by the number of times
            dim_coor = np.array(list(range(self.x_elements_count))) // repeat_times
            # Modulo the dimension length
            dim_coor = dim_coor % self.x_shape[dim]
            self.x_coordinates[dim,] = dim_coor

        # Reshape so that the dimensions after the first one is equal to the shape of x
        self.x_coordinates = np.reshape(self.x_coordinates, newshape=[self.x_dim] + list(self.x_shape))

        Log.debugdebug(
            'Coordinates of x by dimension:\n\r' + str(self.x_coordinates)
        )

        cm = np.zeros(shape=[self.x_dim])
        for dim in range(self.x_dim):
            if np.sum(self.x) > 0:
                cm[dim] = np.sum( self.x_coordinates[dim] * self.x ) / np.sum(self.x)
            else:
                ones_arr = np.ones(shape=self.x_shape)
                cm[dim] = np.sum( self.x_coordinates[dim] *  ones_arr) / np.sum(ones_arr)
        return cm


class CenterOfMassUnitTest:
    def __init__(self, ut_params):
        self.ut_params = ut_params
        return

    def run_unit_test(self):
        res_final = ut.ResultObj(count_ok=0, count_fail=0)

        x = np.array(list(range(0, 50, 1)) + list(range(49, -1, -1)))
        x = np.reshape(x, newshape=(10, 10))
        cm = CenterOfMass(x=x).calculate()
        res = ut.UnitTest.assert_true(
            # Convert to list so can be compared
            observed = cm.tolist(),
            expected = np.array([4.5, 4.5]).tolist(),
            test_comment = 'Test center of mass for symmetrical array ' + str(x) + ', shape ' + str(x.shape)
        )
        res_final.update_bool(res_bool=res)

        x = np.zeros(shape=(10, 5))
        cm = CenterOfMass(x=x).calculate()
        res = ut.UnitTest.assert_true(
            # Convert to list so can be compared
            observed = cm.tolist(),
            expected = np.array([4.5, 2.0]).tolist(),
            test_comment = 'Test center of mass for array of 0s ' + str(x) + ', shape ' + str(x.shape)
        )
        res_final.update_bool(res_bool=res)

        x = np.array([[],[],[]])
        cm = CenterOfMass(x=x).calculate()
        res = ut.UnitTest.assert_true(
            # Need to check the return value [nan, nan] one by one
            observed = (np.isnan(cm[0]) and np.isnan(cm[1])),
            expected = True,
            test_comment = 'Test center of mass for empty array ' + str(x) + ', shape ' + str(x.shape)
        )
        res_final.update_bool(res_bool=res)

        return res_final


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    res = CenterOfMassUnitTest(ut_params=None).run_unit_test()
    # exit(res.count_fail)

    x = np.array(list(range(5*6*7)))
    x = np.reshape(x, newshape=(5,6,7))
    print(x)
    cm = CenterOfMass(x=x).calculate()
    print(cm)

    x = np.array([[]])
    print(CenterOfMass(x=x).calculate())