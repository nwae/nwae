# -*- coding: utf-8 -*-

import nltk
import nltk.stem.wordnet as wordnet
import nltk.stem.porter as porter
import nltk.stem.snowball as snowball
import nwae.utils.Log as lg
from inspect import getframeinfo, currentframe
import nwae.utils.Profiling as prf


class Lemmatizer:

    TYPE_PORTER_STEMMER = 'porter-stemmer'
    TYPE_SNOWBALL_STEMMER = 'snowball-stemmer'
    TYPE_WORDNET_LEMMATIZER = 'wordnet-lemmatizer'

    def __init__(
            self,
            stemmer_type = TYPE_PORTER_STEMMER
    ):
        self.stemmer_type = stemmer_type

        self.stemmer = None

        if self.stemmer_type == Lemmatizer.TYPE_WORDNET_LEMMATIZER:
            nltk.download('wordnet')
            self.stemmer = wordnet.WordNetLemmatizer()
        elif self.stemmer_type == Lemmatizer.TYPE_PORTER_STEMMER:
            self.stemmer = porter.PorterStemmer()
        elif self.stemmer_type == Lemmatizer.TYPE_SNOWBALL_STEMMER:
            self.stemmer = snowball.SnowballStemmer(
                language = 'english'
            )
        else:
            raise Exception(
                'Unrecognized stemmer type "' + str(self.stemmer_type) + '".'
            )

        # Call once, because only the first one is slow
        self.stem(word='initialize')

        return

    def stem(
            self,
            word
    ):
        if self.stemmer_type == Lemmatizer.TYPE_WORDNET_LEMMATIZER:
            return self.stemmer.lemmatize(
                word = word
            )
        else:
            return self.stemmer.stem(
                word = word
            )


if __name__ == '__main__':
    l_lemma = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_WORDNET_LEMMATIZER
    )
    l_porter = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_PORTER_STEMMER
    )
    l_snowball = Lemmatizer(
        stemmer_type = Lemmatizer.TYPE_SNOWBALL_STEMMER
    )

    words = [
        'initialize', 'article', 'leaves', 'is', 'are', 'programming', 'programmer',
        'books', 'downloading', 'downloader', 'eating', 'ate', 'beauty',
        'люблю', 'ем',
        '미친', '나가'
    ]

    print('Word --> Wordnet Lemmatizer, Porter Stemmer, Snowball Stemmer')
    a = prf.Profiling.start()
    for w in words:
        print(
            str(w) + ' --> '
            + str(l_lemma.stem(word=w))
            + ', ' + str(l_porter.stem(word=w))
            + ', ' + str(l_snowball.stem(word=w))
        )

    b = prf.Profiling.stop()
    total_time_secs = prf.Profiling.get_time_dif(start=a, stop=b)
    total_time_ms = total_time_secs * 1000
    rps = round(len(words) / total_time_secs, 4)
    tpr = round(1/rps, 4)
    print('Time start: ' + str(a) + ', time end: ' + str(b))
    print('RPS: ' + str(rps) + 'rps, time per request = ' + str(tpr) + 'ms')

    exit(0)
