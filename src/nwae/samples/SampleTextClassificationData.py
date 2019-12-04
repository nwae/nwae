# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.preprocessing.BasicPreprocessor import BasicPreprocessor

class SampleTextClassificationData:

    COL_CLASS_ID = 'classId'
    COL_CLASS = 'class'
    COL_TEXT = 'text'
    COL_TEXT_ID = 'textId'

    DATA_LANGUAGE = LangFeatures.LANG_KO

    # The class text
    DATA_TEXTS = [
        # 0
        ['하나', '두', '두', '셋', '넷'],
        ['하나', '하나', '두', '셋', '셋', '넷'],
        ['하나', '두', '셋', '넷'],
        # 1
        ['두', '셋', '셋', '넷'],
        ['두', '두', '셋', '셋', '넷', '넷'],
        ['두', '두', '셋', '넷', '넷'],
        # 2
        ['넷', '다섯', '다섯', '여섯', '여섯', '여섯'],
        ['두', '넷', '넷', '다섯', '다섯', '여섯', '여섯'],
        ['두', '넷', '다섯', '여섯', '여섯'],
        # 3
        ['하나', '여섯'],
        ['하나', '여섯', '여섯'],
        ['하나', '하나', '여섯'],
        ['두', '셋', '넷', '다섯'],
        ['두', '셋', '셋', '넷', '다섯'],
        ['두', '셋', '넷', '넷', '다섯']
    ]
    # The class labels
    DATA_Y = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    )

    @staticmethod
    def get_text_classification_training_data():
        word_sep = BasicPreprocessor.get_word_separator(lang=SampleTextClassificationData.DATA_LANGUAGE)
        texts_arr = SampleTextClassificationData.DATA_TEXTS
        texts_str = []
        for txt in texts_arr:
            texts_str.append(word_sep.join(txt))

        return {
            SampleTextClassificationData.COL_CLASS_ID: range(SampleTextClassificationData.DATA_Y.shape[0]),
            SampleTextClassificationData.COL_CLASS: SampleTextClassificationData.DATA_Y,
            SampleTextClassificationData.COL_TEXT: texts_str,
            SampleTextClassificationData.COL_TEXT_ID: range(SampleTextClassificationData.DATA_Y.shape[0])
        }


if __name__ == '__main__':
    data = SampleTextClassificationData.get_text_classification_training_data()
    df = pd.DataFrame(data)
    print(data)
    print(df)
