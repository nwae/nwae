#!/use/bin/python
# --*-- coding: utf-8 --*--

import re
import nwae.lib.lang.LangFeatures as lf


#
# In human communications, words are often "Latinized"
# Instead of "你好" we have "nihao", or "sawusdee" instead of "สวัสดี".
# In Vietnamese, different 'a' forms are simplified to 'a' without diacritics, etc.
#
class LatinEquivalentForm:

    def __init__(self):
        return

    @staticmethod
    def get_latin_equivalent_form(
            # Language is just to speed up the function so that
            # it does not do anything if not required, you may pass None
            lang,
            word
    ):
        wordlatin = word
        # For Vietnamese, we add a Latin column mapping (actually we can also do this for other languages)
        if lang in [lf.LangFeatures.LANG_VI, lf.LangFeatures.LANG_VN]:
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
        else:
            # TODO: Convert to latin equivalent for other languages
            wordlatin = word

        return wordlatin


if __name__ == '__main__':
    lang = lf.LangFeatures.LANG_VN
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lang, word='Anh yêu em'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=lang, word='đây là tiếng Latin'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=None, word='니는 영화를 조아'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=None, word='我喜欢吃点心'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=None, word='как дела'))
    print(LatinEquivalentForm.get_latin_equivalent_form(lang=None, word='สวัสดี ไปไหนมา'))
