# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
import numpy as np
from nwae.lang.corpora.Scrape import Scrape
from nwae.utils.StringUtils import StringUtils
import re
from bs4 import BeautifulSoup
import urllib.parse


class ScrapeUrl:

    def __init__(self):
        return

    def get_training_data_from_file(
            self,
            filepath,
            min_char_per_sent = 0,
            max_char_per_sent = np.inf,
    ):
        f = open(file=filepath, mode='r', encoding='utf-8')
        sentences_list = f.readlines()
        sentences_list = [
            s for s in sentences_list
            if (len(s) >= min_char_per_sent) and (len(s) <= max_char_per_sent)
        ]
        return sentences_list

    def get_training_data_by_scraping_urls(
            self,
            url_list,
            tag_to_find       ='p',
            min_char_per_sent = 0,
            max_char_per_sent = np.inf,
            write_to_filepath = None,
    ):
        sentences_list_agg = []
        for url in url_list:
            sentences_list = self.get_training_data_by_scraping(
                url               = url,
                tag_to_find       = tag_to_find,
                min_char_per_sent = min_char_per_sent,
                max_char_per_sent = max_char_per_sent,
            )
            sentences_list_agg += sentences_list
            # tokens_list_agg += tokens_list
            # is_sep_list_agg += is_sep_list

        if write_to_filepath:
            f = open(file=str(write_to_filepath), mode='w', encoding='utf-8')
            [f.write(str(s) + '\n') for s in sentences_list_agg]
            f.close()

        return sentences_list_agg

    def get_training_data_by_scraping(
            self,
            url,
            tag_to_find       = 'p',
            min_char_per_sent = 0,
            max_char_per_sent = np.inf,
            rm_html_markup    = False,
            unquote_html      = False,
    ):
        # Пример данных из википедии
        sentences_list_from_wiki_scraping = Scrape().scrape_url(
            url=url,
            tag_to_find=tag_to_find
        )
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Scraped ' + str(len(sentences_list_from_wiki_scraping)) + ' sentences from url "' + str(url) + '"'
        )
        sentences_list = []
        for s in sentences_list_from_wiki_scraping:
            s = StringUtils.trim(s)
            s = BeautifulSoup(s).text
            s_clean = s
            if rm_html_markup:
                # Remove all patterns '<...>'
                html_tags_re = re.compile(r'<[^>]+>')
                s_clean = re.sub(html_tags_re, '', string=s)
            if unquote_html:
                # Convert strings like '%3Fmode%3DLSD%26mid%3Dshm%26sid1%3D102%26oid%3D421%26aid%3D0005537039'
                # into '?mode=LSD&mid=shm&sid1=102&oid=421&aid=0005537039'
                s_clean = urllib.parse.unquote(string=s)
            len_s = len(s_clean)
            if (len_s >= min_char_per_sent) and (len_s <= max_char_per_sent):
                sentences_list.append(s_clean)
            Log.debug(
                'From\n\r\t"' + str(s) + '" to\n\r\t"' + str(s_clean) + '"'
            )
        Log.info(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Filtered to ' + str(len(sentences_list)) + ' sentences from url "' + str(url) + '"'
        )
        return sentences_list


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1

    sentences_list = ScrapeUrl().get_training_data_by_scraping_urls(
        url_list=[
            'https://ja.wikipedia.org/wiki/ソニー',
            'https://ja.wikipedia.org/wiki/2020年東京オリンピック',
            'https://ja.wikipedia.org/wiki/新型コロナウイルス感染症の世界的流行_(2019年-)',
            'https://ja.wikipedia.org/wiki/SARSコロナウイルス2',
            'https://ja.wikipedia.org/wiki/新型コロナウイルス感染拡大による東京オリンピック・パラリンピックへの影響',
            'https://slowcook.netlify.app/mix/2050-recipe-of-homemade-nakji-bokkeum-korean-spicy-octopus-stirfry/',
            'https://www.bbc.com/ukrainian/vert-earth-russian-47766544',
            'https://www.say7.info/cook/recipe/118-Plov.html',
            'https://ru.wikipedia.org/wiki/IU_(%D0%BF%D0%B5%D0%B2%D0%B8%D1%86%D0%B0)',
        ],
        tag_to_find='p',
        min_char_per_sent=10,
        max_char_per_sent=30,
        write_to_filepath=None,
    )
    print('***** TOTAL SCRAPED = ' + str(len(sentences_list)))
    # [print(s) for s in sentences_list]
