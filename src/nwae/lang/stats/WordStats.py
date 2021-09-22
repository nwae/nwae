# --*-- coding: utf-8 --*--

import pandas as pd
from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.lang.corpora.ScrapeUrl import ScrapeUrl
from nwae.lang.preprocessing.TxtPreprcsrAllLang import TextPreprscrAllLang


class WordStats:

    def __init__(
            self,
            url_list,
            dirpath_wordlist = None,
            postfix_wordlist = None,
            dirpath_app_wordlist = None,
            postfix_app_wordlist = None,
            dirpath_synonymlist = None,
            postfix_synonymlist = None,
    ):
        self.url_list = url_list
        self.txt_preprocessor = TextPreprscrAllLang(
            dir_wordlist         = dirpath_wordlist,
            postfix_wordlist     = postfix_wordlist,
            dir_app_wordlist     = dirpath_app_wordlist,
            postfix_app_wordlist = postfix_app_wordlist,
            dir_synlist          = dirpath_synonymlist,
            postfix_synlist      = postfix_synonymlist,
            stopwords_list       = [],
        )
        return

    def scrape(self):
        self.sentences_scraped = ScrapeUrl().get_training_data_by_scraping_urls(
            url_list = self.url_list,
            tag_to_find = 'p',
            min_char_per_sent = 0,
            max_char_per_sent = 9999,
            write_to_filepath = None,
        )
        Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Scraped ' + str(len(self.sentences_scraped)) + ' from urls ' + str(self.url_list)
        )

        self.sentences_processed = self.txt_preprocessor.preprocess_list_all_langs(sentences_list=self.sentences_scraped)

    def get_stats(self):
        words_list = []
        for s in self.sentences_processed:
            [words_list.append(w) for w in s]

        df = pd.DataFrame({'Word': words_list, 'Count': 1})
        df = df.groupby(
            by = ['Word']
        ).sum()
        df = df.sort_values(by=['Count'], ascending=False)
        df.to_csv('stats.csv')


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    wstats = WordStats(
        url_list=[
            'https://ru.wikipedia.org/wiki/Криогеника',
            'https://ru.wikipedia.org/wiki/Азот',
        ],

    )
    wstats.scrape()
    wstats.get_stats()

    exit(0)
