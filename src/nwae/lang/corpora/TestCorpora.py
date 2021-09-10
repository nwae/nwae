# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lang.corpora.ScrapeUrl import ScrapeUrl
import numpy as np


class TestCorpora:

    def __init__(self):
        return

    def test_corpora_general(
            self,
            data_from_internet = True,
            write_to_file_path = None,
            sample_fpath       = None,
    ):
        if data_from_internet:
            sentences_list = ScrapeUrl().get_training_data_by_scraping_urls(
                url_list=[
                    'https://slowcook.netlify.app/mix/2050-recipe-of-homemade-nakji-bokkeum-korean-spicy-octopus-stirfry/',
                    'https://www.bbc.com/ukrainian/vert-earth-russian-47766544',
                    'https://www.say7.info/cook/recipe/118-Plov.html',
                    'https://ru.wikipedia.org/wiki/IU_(%D0%BF%D0%B5%D0%B2%D0%B8%D1%86%D0%B0)',
                ],
                tag_to_find       = 'p',
                min_char_per_sent = 50,
                max_char_per_sent = 500,
                write_to_filepath = write_to_file_path,
            )
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': TOTAL SCRAPED = ' + str(len(sentences_list))
            )
        else:
            assert sample_fpath is not None
            sentences_list = ScrapeUrl().get_training_data_from_file(
                filepath = sample_fpath,
                min_char_per_sent = 0,
                max_char_per_sent = np.inf,
            )
            for i in range(len(sentences_list)):
                s = sentences_list[i]
                import re
                sentences_list[i] = re.sub(pattern='&nbsp', repl='', string=s)
            # sentences_list = [s for s in sentences_list if (len(s) >= 10) and (len(s) <= 30)]
            Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': TOTAL READ = ' + str(len(sentences_list))
            )
            # [print(s) for s in sentences_list]

        return sentences_list


if __name__ == '__main__':
    sents = TestCorpora().test_corpora_general()
    [print(s) for s in sents]
    exit(0)

