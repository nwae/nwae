# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from nwae.math.NumpyUtil import NumpyUtil
import keras.utils as kerasutils
import pickle


class Boosting:

    def __init__(
            self,
            model = None,
            model_path = None,
    ):
        self.model = model
        self.model_path = model_path
        if self.model_path is not None:
            self.load_model(path=self.model_path)
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

    def convert_to_xgboost_data_format(
            self,
            data,
            labels,
            feature_names = None,
    ):
        # Data format for xgboost
        if feature_names is None:
            return xgb.DMatrix(
                data  = data,
                label = labels,
            )
        else:
            return xgb.DMatrix(
                data  = data,
                label = labels,
                feature_names = feature_names
            )

    #
    # Implements the generic AdaBoost algorithm on any predictors
    #
    def fit_adaboost(
            self,
            X_train,
            Y_train,
            num_class,
            # List of predictor functions?
            h_pred,
    ):
        raise Exception('not yet implemented')

    #
    # Default xgboost library uses trees as weak learners
    #
    def fit_gradient_boosting(
            self,
            X_train,
            Y_train,
            num_class,
            feature_names,
            num_round = 10,
            # boosting_model 'binary:logistic',
            classtype = 'multi:softprob',
            save_model_path = None,
    ):
        # Convert labels to categorical one-hot encoding
        labels_categorical = kerasutils.to_categorical(Y_train, num_classes=num_class)
        dtrain = self.convert_to_xgboost_data_format(
            data   = X_train,
            labels = Y_train,
            feature_names = feature_names,
        )
        param = {
            'max_depth': 3,
            'eta': 1,
            'objective': classtype,
            'num_class': num_class
        }
        param['nthread'] = 4
        param['eval_metric'] = 'auc'

        # evallist = [(dtest, 'test')]

        self.model = xgb.train(
            param,
            dtrain,
            num_round,
            # evallist
        )
        model_dump = self.model.get_dump()
        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Boosting model class type "' + str(classtype) + '" trained successfully.'
            + ' Number of trees = ' + str(len(model_dump))
            + ', Feature names: ' + str(self.model.feature_names)
        )
        if save_model_path is not None:
            pickle.dump(self.model, open(save_model_path, "wb"))
        return self.model

    def load_model(self, path):
        # Now, load the model for use on a new dataset
        loaded_model = pickle.load(open(path, "rb"))
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Loaded model from file "' + str(path) + '", feature names: ' + str(loaded_model.feature_names)
        )
        return loaded_model

    def predict(
            self,
            X,
            Y,
            feature_names = None
    ):
        dtest = self.convert_to_xgboost_data_format(
            data   = X,
            labels = Y,
            feature_names = feature_names,
        )
        ypred = self.model.predict(dtest)
        return ypred

    def check_prediction_stats(
            self,
            X,
            Y,
            y_predicted,
    ):
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Checking prediction stats..'
        )
        # print(y_predicted)
        # print(type(y_predicted))
        # print(y_predicted.shape)
        # print(np.sum(y_predicted, axis=1).tolist())
        # Compare some data
        count_correct = 0
        for i in range(X.shape[0]):
            data_i = X[i]
            label_i = Y[i]
            prob_distribution = y_predicted[i]
            top_x = NumpyUtil.get_top_indexes(
                data=prob_distribution,
                ascending=False,
                top_x=5
            )
            if top_x[0] == label_i:
                count_correct += 1
            Log.debug(str(i) + '. ' + str(data_i) + ': Label=' + str(label_i) + ', predicted=' + str(top_x))
        Log.important('Boosting Accuracy = ' + str(100 * count_correct / X.shape[0]) + '%.')
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
    features = [str(x) for x in list(range(input_dim))]
    print('***** Start Boost Classifier *****')
    boost.fit_gradient_boosting(
        X_train   = X_train,
        Y_train   = Y_train,
        num_class = num_class,
        feature_names = features,
        save_model_path = 'boost_model.dat'
    )
    model = boost.load_model(path='boost_model.dat')
    boost.check_prediction_stats(
        X = X_test,
        Y = Y_test,
        y_predicted = boost.predict(X = X_test, Y = Y_test, feature_names = features)
    )

    print('Plotting tree..' + str(model.get_dump()))
    plot_tree(model, num_trees=0)
    plt.show()
    print('Done')

    exit(0)
