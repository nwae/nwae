
import threading
import time
import mozg.common.util.Log as log
from inspect import currentframe, getframeinfo


#
# Loads files/data required for intent detection in the background.
# This allows fast server startup without being held back by the loadings.
# It is extremely critical when running in a WSGI gunicorn environment that will
# keep restarting the Flask servers because of slow startup.
#
class IntentEngineThread(threading.Thread):

    SLEEP_TIME = 2
    COUNT_TIME_CHECK_READ_ONLY_HASHES = 20

    def __init__(
            self,
            # Callback IntentEngine Class
            botkey,
            intent_self,
            check_hashes = False
    ):
        super(IntentEngineThread, self).__init__()
        self.stoprequest = threading.Event()

        self.botkey = botkey
        self.intent_self = intent_self
        self.check_hashes = check_hashes
        # We need this flag because threads don't throw exceptions to the caller of the IntentEngine class
        # Thus the caller will not know if exception has occurred.
        self.failed_to_load = False

    def is_failed_to_load(self):
        return self.failed_to_load

    def join(self, timeout=None):
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Botkey "' + str(self.botkey) + '" join called..')
        self.stoprequest.set()
        super(IntentEngineThread, self).join(timeout=timeout)
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Botkey "' + str(self.botkey) + '" Intent Background Thread ended..')

    def __start_intent_engine(self):
        try:
            # Callback the IntentEngine class function
            self.intent_self.background_load_rfv_commands_from_file()
            log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                             + ': Botkey "' + str(self.botkey) + '" Background load done.')
            # if self.intent_self.reduce_features:
            #     log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #                      + ': Botkey "' + str(self.botkey) + '" Reducing features.')
            #     self.intent_self.reduce_and_optimize_features()
            #     log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            #                      + ': Botkey "' + str(self.botkey) + '" Reducing features done.')

            self.intent_self.set_rfv_to_ready()
            self.intent_self.set_training_data_to_ready()

            self.intent_self.get_hash_of_readonly_objects()
        except Exception as ex:
            self.failed_to_load = True
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Botkey "' + str(self.botkey)\
                     + '" Exception in background thread rfv/etc from file. ' + str(ex)
            log.Log.critical(errmsg)
            # Don't throw exception here, no one is catching it as it is a thread
            # raise Exception(errmsg)

    def run(self):
        log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                         + ': Botkey "' + str(self.botkey) + '" Intent Background Thread started..')
        sleep_time = 10
        while True:
            if self.stoprequest.isSet():
                log.Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Botkey "' + str(self.botkey) + '" Breaking from forever thread...'
                )
                break
            if self.intent_self.check_if_rfv_updated():
                self.__start_intent_engine()
                if self.failed_to_load:
                    log.Log.important(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Intent Engine failed to load. Try again in ' + str(sleep_time) + ' secs..'
                    )
            time.sleep(sleep_time)

        #
        # Checking hashes are slow, do it only if necessary
        #
        # count = 0
        # while True:
        #     # Use modulo count
        #     count = (count + 1) % IntentEngineThread.COUNT_TIME_CHECK_READ_ONLY_HASHES
        #
        #     if self.stoprequest.isSet():
        #         log.Log.critical(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #                          + ': Received stop request...')
        #         break
        #
        #     if count == 0:
        #         try:
        #             self.intent_self.check_hash_of_readonly_objects()
        #             log.Log.important(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #                          + ': Intent Engine Hashes OK after '
        #                          + str(self.intent_self.get_count_intent_calls()) + ' function calls.')
        #         except Exception as ex:
        #             errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
        #                      + ': Calls to intent class = ' + str(self.intent_self.get_count_intent_calls()) \
        #                      + '. Exception ' + str(ex) + '.'
        #             log.Log.critical(errmsg)
        #     else:
        #         log.Log.debugdebug(str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
        #                            + ': Nothing to do. count = ' + str(count) + '.')
        #
        #     time.sleep(IntentEngineThread.SLEEP_TIME)
