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

        self.x_shape = self.x.shape
        # How many elements altogether
        self.x_len = np.product(self.x_shape)
        self.x_dim = len(self.x_shape)

        self.x_coordinates = np.zeros(shape=[self.x_dim] + [self.x_len])

        repeat_times = self.x_len
        for dim in range(self.x_dim):
            repeat_times = repeat_times / self.x_shape[dim]
            dim_coor = np.array(list(range(self.x_len))) // repeat_times
            # Modulo the dimension length
            dim_coor = dim_coor % self.x_shape[dim]
            self.x_coordinates[dim,] = dim_coor

        # Reshape so that the dimensions after the first one is equal to the shape of x
        self.x_coordinates = np.reshape(self.x_coordinates, newshape=[self.x_dim] + list(self.x_shape))

        Log.debugdebug(
            'Coordinates of x by dimension:\n\r' + str(self.x_coordinates)
        )

        return

    def calculate(
            self
    ):
        cm = np.zeros(shape=[self.x_dim])
        for dim in range(self.x_dim):
            cm[dim] = np.sum( self.x_coordinates[dim] * self.x ) / np.sum(self.x)
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
            test_comment = 'Test center of mass for symmetrical array ' + str(x)
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
