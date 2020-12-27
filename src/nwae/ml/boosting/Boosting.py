# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
import xgboost as xgb
from nwae.math.NumpyUtil import NumpyUtil
import keras.utils as kerasutils


class Boosting:

    def __init__(
            self,
    ):
        return

    def generate_random_data(
            self,
            n,
            input_dim,
            test_prop=0.2
    ):
        #
        # Prepare random data
        #
        # Random vectors numpy ndarray type
        X_train = np.random.random((n, input_dim))
        #
        # Design some pattern
        # Labels are sum of the rows, then floored to the integer
        # Sum >= 0, 1, 2, 3,...
        #
        row_sums = np.sum(X_train, axis=1)
        Y_train = np.array(np.round(row_sums - 0.5, 0), dtype=int)

        # Split to test/train
        cut_off = int((1 - test_prop) * n)

        X_test = X_train[cut_off:n]
        Y_test = Y_train[cut_off:n]
        X_train = X_train[0:cut_off]
        Y_train = Y_train[0:cut_off]

        # labels = np.random.randint(n_labels, size=(n_rows, 1))

        # Print some data
        for i in range(10):
            print(str(i) + '. ' + str(Y_train[i]) + ': ' + str(X_train[i]))

        return X_train, Y_train, X_test, Y_test

    def create_boosting_model(
            self,
            data,
            labels
    ):
        return xgb.DMatrix(
            data=data,
            label=labels
        )

    def classify_boosting(
            self,
            X_train,
            Y_train,
            X_test,
            Y_test,
            num_class
    ):
        # Convert labels to categorical one-hot encoding
        labels_categorical = kerasutils.to_categorical(Y_train, num_classes=num_class)
        dtrain = self.create_boosting_model(
            data=X_train,
            labels=Y_train
        )
        dtest = self.create_boosting_model(
            data=X_test,
            labels=Y_test
        )
        param = {
            'max_depth': 3,
            'eta': 1,
            # 'objective': 'binary:logistic',
            'objective': 'multi:softprob',
            'num_class': num_class
        }
        param['nthread'] = 4
        param['eval_metric'] = 'auc'

        evallist = [(dtest, 'test')]

        num_round = 10
        bst = xgb.train(
            param,
            dtrain,
            num_round,
            # evallist
        )

        ypred = bst.predict(dtest)
        print(ypred)
        print(type(ypred))
        print(ypred.shape)
        print(np.sum(ypred, axis=1).tolist())
        # Compare some data
        count_correct = 0
        for i in range(X_test.shape[0]):
            data_i = X_test[i]
            label_i = Y_test[i]
            prob_distribution = ypred[i]
            top_x = NumpyUtil.get_top_indexes(
                data=prob_distribution,
                ascending=False,
                top_x=5
            )
            if top_x[0] == label_i:
                count_correct += 1
            Log.debug(str(i) + '. ' + str(data_i) + ': Label=' + str(label_i) + ', predicted=' + str(top_x))
        Log.important('Boosting Accuracy = ' + str(100 * count_correct / X_test.shape[0]) + '%.')
        return


if __name__ == '__main__':
    boost = Boosting()
    input_dim = 5
    X_train, Y_train, X_test, Y_test = boost.generate_random_data(
        n=10000,
        input_dim=input_dim
    )
    num_class = len(np.unique(Y_train))
    print(num_class)

    Log.LOGLEVEL = Log.LOG_LEVEL_INFO
    print('***** Start Boost Classifier *****')
    boost.classify_boosting(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, num_class=num_class)

    exit(0)
