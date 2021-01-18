#!/use/bin/python
# --*-- coding: utf-8 --*--

# !!! Will work only on Python 3 and above

import re
import nwae.utils.FileUtils as futil
import nwae.utils.StringUtils as su
from nwae.lang.LangFeatures import LangFeatures
import nwae.lang.nlp.LatinEquivalentForm as lef
import nwae.lang.characters.LangCharacters as langchar
import nwae.utils.Log as log
from inspect import getframeinfo, currentframe


class SynonymList:

    def __init__(
            self,
            lang,
            dirpath_synonymlist,
            postfix_synonymlist,
            add_latin_equiv_words = False
    ):
        self.lang = LangFeatures.map_to_lang_code_iso639_1(
            lang_code = lang
        )

        self.dirpath_synonymlist = dirpath_synonymlist
        self.postfix_synonymlist = postfix_synonymlist
        self.add_latin_equiv_words = add_latin_equiv_words

        self.map_word_to_rootword = {}
        return

    def load_synonymlist(
            self,
            # List of words that should be in the first position (index 0)
            allowed_root_words = None
    ):
        self.map_word_to_rootword = self.__load_list(
            dirpath = self.dirpath_synonymlist,
            postfix = self.postfix_synonymlist,
            allowed_root_words = allowed_root_words
        )
        return

    # General function to load wordlist or stopwords
    def __load_list(
            self,
            dirpath,
            postfix,
            allowed_root_words = None
    ):
        filepath = str(dirpath) + '/' + str(self.lang) + str(postfix)
        try:
            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Loading list for lang "' + self.lang + '" from file path "' + filepath + '"'
            )

            fu = futil.FileUtils()
            content = fu.read_text_file(
                filepath = filepath,
                throw_exception = True,
            )
        except Exception as file_ex:
            content = []
            errmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno) \
                     + ': Error loading synonym list for lang "' + str(self.lang) \
                     + '", directory path "' + str(dirpath) + '", postfix "' + str(postfix) \
                     + '" from file "' + str(filepath) + '": ' + str(file_ex)
            log.Log.error(errmsg)

        log.Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Synonym Read ' + str(len(content)) + ' lines.'
            + ' List of main words (' + str(type(allowed_root_words)) + '):\n\r' + str(allowed_root_words)
        )

        map_word_rootword = {}

        for line in content:
            line = str(line)
            line = su.StringUtils.trim(line)
            # Remove empty lines
            if len(line)<=0: continue
            # Remove comment lines starting with '#'
            if re.match(pattern='^#', string=line): continue

            linewords = line.split(sep=',')
            if type(linewords) not in (list, tuple):
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Line "' + str(linewords) + '" not split into list or tuple but type '
                    + str(type(linewords)) + '.'
                )
                continue
            # Trim and convert to lowercase
            linewords = [su.StringUtils.trim(str(x).lower()) for x in linewords]
            # Remove empty string
            linewords = [x for x in linewords if x]
            if len(linewords) == 0:
                continue
            # Add latin equivalent forms here
            if self.add_latin_equiv_words:
                linewords_lef = [
                    lef.LatinEquivalentForm.get_latin_equivalent_form(lang=self.lang, word=x) for x in linewords
                ]
                for w in linewords_lef:
                    if w not in linewords:
                        linewords.append(w)

            log.Log.info(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Processing line "' + str(linewords) + '".'
            )
            rootword = None
            for rootword_test in linewords:
                if allowed_root_words is not None:
                    if rootword_test not in allowed_root_words:
                        log.Log.warning(
                            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                            + ': Word "' + str(rootword_test) + '" not in allowed root words! Trying next word..'
                        )
                        continue
                # root word is in list of main words, or list main words is None
                rootword = rootword_test
                break

            # If no allowed rootwords found, ignore entire line
            if rootword is None:
                log.Log.warning(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Line "' + str(linewords) + '" ignored, no root words found!'
                )
                continue

            log.Log.debugdebug(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Line "' + str(linewords) + '" root word "' + str(rootword) + '"'
            )

            for word in linewords:
                if word in map_word_rootword.keys():
                    warnmsg = str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)\
                              + ': Duplicate word "' + str(word) + '" in synonym list!'
                    log.Log.warning(warnmsg)
                map_word_rootword[word] = rootword

        log.Log.info(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Successfully loaded synonym list:\n\r' + str(map_word_rootword)
        )
        return map_word_rootword

    def get_synonym_list_words(self):
        return list(self.map_word_to_rootword.keys())

    # Replace with root words, thus normalizing the text
    def normalize_text_array(
            self,
            # A word array. e.g. ['this','is','a','sentence','or','just','any','word','array','.']
            text_segmented_array
    ):
        words_normalized = []
        #
        # Replace words with root words
        #
        wordkeys = self.map_word_to_rootword.keys()
        for word in text_segmented_array:
            if not word:
                continue

            if word in wordkeys:
                rootword = self.map_word_to_rootword[word]
                words_normalized.append(rootword)
            else:
                words_normalized.append(word)

        return words_normalized


if __name__ == '__main__':
    import nwae.config.Config as cf
    config = cf.Config.get_cmdline_params_and_init_config_singleton(
        Derived_Class       = cf.Config,
        default_config_file = cf.Config.CONFIG_FILE_PATH_DEFAULT
    )

    for lang in [LangFeatures.LANG_ZH, LangFeatures.LANG_TH, LangFeatures.LANG_VI]:
        sl = SynonymList(
            lang                = lang,
            dirpath_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_DIR_SYNONYMLIST),
            postfix_synonymlist = config.get_config(param=cf.Config.PARAM_NLP_POSTFIX_SYNONYMLIST)
        )
        sl.load_synonymlist()

        sentences = [
            ['试试','这个','提钱','入钱','代理','好','พม','คับ','คะ', 'alo', 'dm', 'con cac']
        ]
        for s in sentences:
            print(sl.normalize_text_array(text_segmented_array=s))

