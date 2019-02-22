#!/usr/bin/python
# -*- coding: utf-8 -*-

import ie.lib.util.StringUtils as su
import re


class CommandLine:

    @staticmethod
    def get_user_filename():
        ui = None
        while True:
            ui = input('Enter Filename (\'m\' to return to Main Menu): ')
            if ui == 'm':
                ui = None
                break
            elif su.StringUtils.trim(ui).__len__() == 0:
                print('File name is empty!')
                continue
            else:
                break
        return ui

    @staticmethod
    def get_user_input_language():
        ui_lang = None
        while True:
            print('Pick Language:')
            print('  1: CNY')
            print('  2: THB')
            print('  (Coming soon VND)')
            print('  m: Back to Main Menu')
            print('                ')
            ui_lang = input('Enter Language: ')
            if su.StringUtils.trim(ui_lang.lower()) == 'cny' or ui_lang == '1':
                ui_lang = 'cn'
                break
            elif su.StringUtils.trim(ui_lang.lower()) == 'thb' or ui_lang == '2':
                ui_lang = 'th'
                break
            elif ui_lang == 'm':
                ui_lang = None
                break
            else:
                print('Invalid choice [' + ui_lang + ']')
        return ui_lang

    @staticmethod
    def get_user_input_brand():
        ui_brand = None
        while True:
            print('Pick Brand:')
            print('  1: Betway')
            print('  2: Fun88')
            print('  3: TLC')
            print('  4: TBet')
            print('  m: Back to Main Menu')
            print('                ')
            ui_brand = input('Enter Brand: ')
            if su.StringUtils.trim(ui_brand.lower()) == 'betway' or ui_brand == '1':
                ui_brand = 'betway'
                break
            elif su.StringUtils.trim(ui_brand.lower()) == 'fun88' or ui_brand == '2':
                ui_brand = 'fun88'
                break
            elif su.StringUtils.trim(ui_brand.lower()) == 'tlc' or ui_brand == '3':
                ui_brand = 'tlc'
                break
            elif su.StringUtils.trim(ui_brand.lower()) == 'tbet' or ui_brand == '4':
                ui_brand = 'tbet'
                break
            elif ui_brand == 'm':
                ui_brand = None
                break
            else:
                print('Invalid choice [' + ui_brand + ']')
        return ui_brand

    @staticmethod
    def get_user_date(str):
        ui_date = None
        while True:
            ui_date = input(str)
            m = re.search(pattern='[0-9]{4}-[01][0-9]-[0123][0-9]', string=ui_date)
            if m:
                break
            else:
                print('Invalid format for date (YYYY-MM-DD) [' + ui_date + ']')
        return ui_date
