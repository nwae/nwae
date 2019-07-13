#!/usr/bin/python
# -*- coding: utf-8 -*-

import re


class StringUtils(object):

    def __init__(self):
        return

    @staticmethod
    def trim(str):
        # '\s' includes spaces and newlines already
        s = re.sub("\s+$", "", re.sub("^\s+", "", str))
        return s

    @staticmethod
    def remove_newline(str, replacement=' '):
        s = re.sub('\n|\r', replacement, str)
        return s


def demo_1():
    s = "  Privet Mir!  \n\r"
    print('[' + s + ']')

    # Demonstrating that newline is also removed
    ss = StringUtils.trim(s)
    print('[' + ss + ']')
    return


# demo_1()
