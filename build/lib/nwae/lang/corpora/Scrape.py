import requests
# pip install beautifulsoup4
from bs4 import BeautifulSoup
from nwae.utils.StringUtils import StringUtils
from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe


class Scrape:

    def __init__(self):
        return

    def scrape_url(
            self,
            url,
            parser = 'html.parser',
            tag_to_find = 'p',
    ):
        try:
            sents = []
            resp = requests.get(
                url = url,
            )
            soup = BeautifulSoup(resp.content, parser)
            contents_tag = soup.find_all(tag_to_find)
            for cont in contents_tag:
                txt = StringUtils.trim(cont.get_text())
                sent_list = txt.split('。')
                sent_list = [StringUtils.trim(s) for s in sent_list if s]
                if len(sent_list):
                    sents += sent_list
                Log.debug('Split "' + str(txt) + '" into:' + str(sent_list))
                # [Log.debug('\t"' + str(s) + '"') for s in sent_list]

            return sents
        except Exception as ex:
            Log.error(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Error scraping url "' + str(url) + '", exception: ' + str(ex)
            )


if __name__ == '__main__':
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_1
    ws = Scrape()

    sents = ws.scrape_url(
        url='https://ja.wikipedia.org/wiki/ソニー',
    )
    print(sents)