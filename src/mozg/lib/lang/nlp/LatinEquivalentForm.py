#!/use/bin/python
# --*-- coding: utf-8 --*--

import re


#
# In human communications, words are often "Latinized"
# Instead of "你好" we have "nihao", or "sawusdee" instead of "สวัสดี".
# In Vietnamese, different 'a' forms are simplified to 'a' without diacritics, etc.
#
class LatinEquivalentForm:

    def __init__(self):
        return

    @staticmethod
    def get_latin_equivalent_form(lang, word):

        wordlatin = word

        # For Vietnamese, we add a Latin column mapping (actually we can also do this for other languages)
        if lang == 'vn':
            # Map [ăâ àằầ ảẳẩ ãẵẫ áắấ ạặậ] to latin 'a', [ê èề ẻể ẽễ éế ẹệ] to 'e', [ì ỉ ĩ í ị] to 'i',
            # [ôơ òồờ ỏổở õỗỡ óốớ ọộợ] to 'o', [ư ùừ ủử ũữ úứ ụự] to u, [đ] to 'd'
            wordlatin = re.sub(u'[ăâàằầảẳẩãẵẫáắấạặậ]', u'a', wordlatin)
            wordlatin = re.sub(u'[ĂÂÀẰẦẢẲẨÃẴẪÁẮẤẠẶẬ]', u'A', wordlatin)
            wordlatin = re.sub(u'[êèềẻểẽễéếẹệ]', u'e', wordlatin)
            wordlatin = re.sub(u'[ÊÈỀẺỂẼỄÉẾẸỆ]', u'E', wordlatin)
            wordlatin = re.sub(u'[ìỉĩíị]', u'i', wordlatin)
            wordlatin = re.sub(u'[ÌỈĨÍỊ]', u'I', wordlatin)
            wordlatin = re.sub(u'[ôơòồờỏổởõỗỡóốớọộợ]', u'o', wordlatin)
            wordlatin = re.sub(u'[ÔƠÒỒỜỎỔỞÕỖỠÓỐỚỌỘỢ]', u'O', wordlatin)
            wordlatin = re.sub(u'[ưùừủửũữúứụự]', u'u', wordlatin)
            wordlatin = re.sub(u'[ƯÙỪỦỬŨỮÚỨỤỰ]', u'U', wordlatin)
            wordlatin = re.sub(u'[đ]', u'd', wordlatin)
            wordlatin = re.sub(u'[Đ]', u'D', wordlatin)
        else:
            # TODO: Convert to latin equivalent for other languages
            wordlatin = word

        return wordlatin

def demo_1():
    return


demo_1()
