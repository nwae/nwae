# -*- coding: utf-8 -*-

import threading
import time
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo


#
# Interfaces that a Model must implement
#
class ModelInterface(threading.Thread):

    # Terms for dataframe, etc.
    TERM_CLASS    = 'class'
    TERM_SCORE    = 'score'
    TERM_DIST     = 'dist'
    TERM_DISTNORM = 'distnorm'
    TERM_RADIUS   = 'radius'

    # Matching
    MATCH_TOP = 10

    def __init__(
            self,
            identifier_string
    ):
        super(ModelInterface, self).__init__()

        self.identifier_string = identifier_string

        self.stoprequest = threading.Event()

        self.__mutex_load_model = threading.Lock()
        return

    def join(self, timeout=None):
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" join called..'
        )
        self.stoprequest.set()
        super(ModelInterface, self).join(timeout=timeout)
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" Background Thread ended..'
        )

    def run(self):
        log.Log.critical(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Model Identifier "' + str(self.identifier_string) + '" Background Thread started..'
        )
        sleep_time = 10
        while True:
            if self.stoprequest.isSet():
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Model Identifier "' + str(self.identifier_string) + '" Breaking from forever thread...'
                )
                break
            if self.check_if_model_updated():
                try:
                    self.__mutex_load_model.acquire()
                    self.load_model_parameters()
                    if not self.is_model_ready():
                        log.Log.important(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Model "' + self.identifier_string
                            + '" failed to load. Try again in ' + str(sleep_time) + ' secs..'
                        )
                finally:
                    self.__mutex_load_model.release()
            time.sleep(sleep_time)

    def get_model_features(
            self
    ):
        return None

    def predict_classes(
            self,
            # ndarray type of >= 2 dimensions
            x):
        return

    def predict_class(
            self,
            # ndarray type of >= 2 dimensions, single point/row array
            x
    ):
        return

    def train(
            self
    ):
        return

    def load_model_parameters(
            self
    ):
        return

    def is_model_ready(
            self
    ):
        return True

    def check_if_model_updated(
            self
    ):
        return False