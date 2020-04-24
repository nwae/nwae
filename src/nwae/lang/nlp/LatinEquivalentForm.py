#!/use/bin/python
# --*-- coding: utf-8 --*--

import re
import nwae.lang.LangFeatures as lf


#
# In human communications, words are often "Latinized"
# Instead of "你好" we have "nihao", or "sawusdee" instead of "สวัสดี".
# In Vietnamese, different 'a' forms are simplified to 'a' without diacritics, etc.
#
class LatinEquivalentForm:

    def __init__(self):
        return

    @staticmethod
    def have_latin_equivalent_form(
            lang
    ):
        lang_std = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        return lang_std in [
            lf.LangFeatures.LANG_VI,
            lf.LangFeatures.LANG_VN
        ]

    @staticmethod
    def get_latin_equivalent_form(
            # Language is just to speed up the function so that
            # it does not do anything if not required, you may pass None
            lang,
            word
    ):
        lang_std = lf.LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )
        if lang_std in [lf.LangFeatures.LANG_VI, lf.LangFeatures.LANG_VN]:
            return LatinEquivalentForm.get_latin_equivalent_form_vietnamese(word=word)
        else:
            return word

    @staticmethod
    def get_latin_equivalent_form_vietnamese(
            word
    ):
        wordlatin = word
        # Map [ăâ àằầ ảẳẩ ãẵẫ áắấ ạặậ] to latin 'a', [ê èề ẻể ẽễ éế ẹệ] to 'e', [ì ỉ ĩ í ị] to 'i',
        # [ôơ òồờ ỏổở õỗỡ óốớ ọộợ] to 'o', [ư ùừ ủử ũữ úứ ụự] to u, [đ] to 'd'
        wordlatin = re.sub('[ăâàằầảẳẩãẵẫáắấạặậ]', 'a', wordlatin)
        wordlatin = re.sub('[ĂÂÀẰẦẢẲẨÃẴẪÁẮẤẠẶẬ]', 'A', wordlatin)
        wordlatin = re.sub('[êèềẻểẽễéếẹệ]', 'e', wordlatin)
        wordlatin = re.sub('[ÊÈỀẺỂẼỄÉẾẸỆ]', 'E', wordlatin)
        wordlatin = re.sub('[ìỉĩíị]', 'i', wordlatin)
        wordlatin = re.sub('[ÌỈĨÍỊ]', 'I', wordlatin)
        wordlatin = re.sub('[ôơòồờỏổởõỗỡóốớọộợ]', 'o', wordlatin)
        wordlatin = re.sub('[ÔƠÒỒỜỎỔỞÕỖỠÓỐỚỌỘỢ]', 'O', wordlatin)
        wordlatin = re.sub('[ưùừủửũữúứụự]', 'u', wordlatin)
        wordlatin = re.sub('[ƯÙỪỦỬŨỮÚỨỤỰ]', 'U', wordlatin)
        wordlatin = re.sub('[đ]', 'd', wordlatin)
        wordlatin = re.sub('[Đ]', 'D', wordlatin)
        wordlatin = re.sub('[ýỳỷỹỵ]', 'y', wordlatin)

        return wordlatin


if __name__ == '__main__':
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_VI, word='Anh yêu em'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_VI, word='đây là tiếng Latin'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_KO, word='니는 영화를 조아'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_ZH, word='我喜欢吃点心'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_RU, word='как дела'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lf.LangFeatures.LANG_TH, word='สวัสดี ไปไหนมา'))
