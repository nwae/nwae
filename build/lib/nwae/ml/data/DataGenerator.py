# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    # Разделить данные на тренинговые и тестовые
    @staticmethod
    def split_into_train_and_test_data(
            # pandas dataframe, or numpy ndarray
            x,
            y,
            # Default 10%
            test_size = 0.1
    ):
        assert len(x)==len(y)
        n = round(len(x) * (1 - test_size))
        X_train, y_train = x[0:n], y[0:n]
        X_test, y_test = x[n:], y[n:]
        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def generate_seq(
            seq_length,
            min_value    = None,
            max_value    = None,
            # random, repeat
            seq_type     = 'random',
            # Only if sequence type == 'repeat'
            repeat_list  = None,
            # uniform, normal
            distribution = 'uniform',
            number_type  = 'float'
    ):
        rand_seq = None

        try:
            val_range = max_value - min_value
        except:
            val_range = 1.0
            min_value = 0.0
            max_value = 1.0

        if number_type == 'int':
            max_value += 1
            val_range += 1

        if seq_type == 'random':
            if distribution == 'uniform':
                rand_seq = np.random.rand(seq_length)*val_range + min_value
            elif distribution == 'normal':
                rand_seq = np.random.normal(0,1,seq_length)
            else:
                raise Exception('Unsupported distribution "' + str(distribution) + '"')
        elif seq_type == 'repeat':
            rep_times = int(1 + seq_length / len(repeat_list))
            data = np.array(repeat_list * rep_times)
            rand_seq = data[0:seq_length]

        if number_type == 'int':
            rand_seq = np.floor(rand_seq)

        return rand_seq


if __name__ == '__main__':
    SHOW_PLOT = False

    # Integral values
    data = DataGenerator.generate_seq(
        seq_length  = 1000,
        min_value   = 1,
        max_value   = 10,
        number_type = 'int'
    )
    print(data)
    print('Mean = ' + str(data.mean()))
    for i in range(1,11,1):
        prob = np.sum(1*(data==i)) / len(data)
        print('Prob ' + str(i) + ' = ' + str(round(prob,2)))

    for distribution in ['uniform', 'normal']:
        data = DataGenerator.generate_seq(
            seq_length   = 1000,
            distribution = distribution
        )
        print(data)
        print(
            'Mean=' + str(np.mean(data))
            + ', Median=' + str(np.median(data))
            + ', Var=' + str(np.var(data))
        )
        (x_train, y_train), (x_test, y_test) = DataGenerator.split_into_train_and_test_data(
            x = data,
            y = list(range(len(data)))
        )
        print('x train shape: ' + str(x_train.shape))
        print('x test shape: ' + str(x_test.shape))
        if SHOW_PLOT:
            plt.hist(data, density=True, bins=50)
            plt.show()

    repeat_list = (list(range(1, 10, 1)) + list(range(10, 1, -1)))
    print('Generate repeat list from : ' + str(repeat_list))
    data = DataGenerator.generate_seq(
        seq_length  = 100,
        seq_type    = 'repeat',
        repeat_list = repeat_list
    )
    print(data)
    (x_train, y_train), (x_test, y_test) = DataGenerator.split_into_train_and_test_data(
        x = data,
        y = list(range(len(data)))
    )
    print('x train shape: ' + str(x_train.shape))
    print('x test shape: ' + str(x_test.shape))


