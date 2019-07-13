# -*- coding: utf-8 -*-

import datetime as dt
import time
import ie.lib.util.StringUtils as su
import os


class Log:
    LOGFILE = ''
    DEBUG_PRINT_ALL_TO_SCREEN = False

    def __init__(self, logfile):
        self.logfile = logfile
        return

    @staticmethod
    def set_path(logfile_path):
        Log.LOGFILE = logfile_path

    @staticmethod
    def log(s, encoding='utf-8'):
        if s is None:
            return

        # Because sometimes we just dump whole objects to log
        s = str(s)

        s = su.StringUtils.trim(str=s)
        if len(s) == 0:
            return

        if Log.LOGFILE == '' or Log.DEBUG_PRINT_ALL_TO_SCREEN:
            print(s)
            return

        try:
            f = None
            if os.path.isfile(Log.LOGFILE):
                f = open(file=Log.LOGFILE, mode='a', encoding=encoding)
            else:
                f = open(file=Log.LOGFILE, mode='w', encoding=encoding)

            timestamp = dt.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            f.write(timestamp + ': ' + s + '\n')
            f.close()
        except Exception as ex:
            errmsg = 'Log file [' + Log.LOGFILE + '] don''t exist!'
            print(errmsg)
            raise(ex)
