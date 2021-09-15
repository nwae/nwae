# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import currentframe, getframeinfo
from nwae.lang.detect.LangDetect import LangDetect
from nwae.lang.LangFeatures import LangFeatures
try:
    from nltk import pos_tag, word_tokenize, download
    download('punkt')
    download('averaged_perceptron_tagger')
except Exception as ex:
    raise Exception(
        str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
        + ': Unable to load nltk pos_tag: ' + str(ex)
    )


"""
Вместо этих "уродливых стоп-слов", у нас такой подход
  - через частеречную разметку, присвоить вес каждому слову в зависимости от типа слова
  - вес всех разметок предопределен. например, маленкое значение частице речи но большое
    значение имя-существительному
"""
class Stopwordtags:

    # https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
    DEFAULT_KEEP_TAGS_ENG = ('JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ')
    DEFAULT_DISCARD_TAGS_ENG = ('CC', 'CD', 'DT', 'EX', 'IN', 'MD', 'PDT', 'TO')

    # TODO Expand
    # 名詞=noun, 動詞=verb
    DEFAULT_KEEP_TAGS_JAP = ('名詞', '動詞')
    # 助詞=particle, 接頭辞=prefix, 補助記号=auxiliary symbol
    DEFAULT_DISCARD_TAGS_JAP = ('助詞', '接頭辞', '補助記号')

    def __init__(
            self,
    ):
        return

    def filter_sentence_by_pos_tag_english(
            self,
            word_list,
            keep_tags = DEFAULT_KEEP_TAGS_ENG,
    ):
        if type(word_list) is str:
            word_list = word_tokenize(text=word_list, language='english')
        words_postags = pos_tag(word_list)
        sent_filtered = [w for w,t in words_postags if (t in keep_tags)]
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': POS TAGs: ' + str(words_postags)
        )
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Filtered sentence: ' + str(sent_filtered)
        )
        return sent_filtered

    def filter_sentence_by_pos_tag_japanese(
            self,
            # string or word list
            word_list,
            keep_tags = DEFAULT_KEEP_TAGS_JAP,
    ):
        try:
            import nagisa
        except Exception as ex:
            raise Exception(
                str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Unable to load nagisa: ' + str(ex)
            )
        if type(word_list) in [list, tuple]:
            text = ' '.join(word_list)
        else:
            text = word_list
        words_postags_obj = nagisa.tagging(text)
        txt_sym_tok = words_postags_obj.words
        txt_sym_postags = words_postags_obj.postags
        Log.debug(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Japanese segmentation ' + str(txt_sym_tok) + ', word & POS tags: ' + str(txt_sym_postags)
        )

        words_postags = list(zip(txt_sym_tok, txt_sym_postags))
        sent_filtered = [w for w,t in words_postags if (t in keep_tags)]
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': POS TAGs: ' + str(words_postags)
        )
        Log.debugdebug(
            str(__name__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Filtered sentence: ' + str(sent_filtered)
        )
        return sent_filtered


if __name__ == '__main__':
    Log.LOGLEVEL = Log.LOG_LEVEL_DEBUG_2
    Log.DEBUG_PRINT_ALL_TO_SCREEN = True
    sents = [
        'Freezing temperatures have gripped the nation, making Wednesday the coldest day yet this winter.',
        'Morning lows plunged to minus 16-point-three degrees Celsius in Seoul , the lowest to be posted during this year’s cold season.',
        'As of 7 a.m. Wednesday , morning lows stood at minus 15-point-four degrees in Daejeon , nearly minus 22 degrees in the Daegwallyeong mountain pass in Pyeongchang and minus 14 degrees in Gangneung.',
        'Due to the wind chill factor, temperatures stood at nearly minus 23 degrees in Seoul , minus 25 in Incheon and roughly minus 36 degrees in Daegwallyeong .',
        'An official of the Korea Meteorological Administration said the nation will continue to see subzero temperatures for the time being with the central regions and some southern inland areas projected to see morning lows plunge below minus 15 degrees',
        'Currently , a cold wave warning is in place for Seoul , Incheon , Daejeon and Sejong as well as the provinces of Gangwon , Chungcheong , North Jeolla and North Gyeongsang.',
        '本日はチャットサービスをご利用いただき、ありがとうございます。オペレーターと接続中です。',
        'The code run is successful',
        'I run the code',
        'Run',
    ]
    ld = LangDetect()
    swt = Stopwordtags()
    text_tag = []
    for sent in sents:
        lang = ld.detect(text = sent)
        if lang:
            lang = lang[0]
        else:
            continue

        if lang == LangFeatures.LANG_EN:
            sent_new = swt.filter_sentence_by_pos_tag_english(word_list=sent)
        elif lang == LangFeatures.LANG_JA:
            sent_new = swt.filter_sentence_by_pos_tag_japanese(word_list=sent)
        else:
            raise Exception(lang)
        text_tag.append(sent_new)
        print()
    exit(0)


"""
From https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk

CC: conjunction, coordinating
& 'n and both but either et for less minus neither nor or plus so
therefore times v. versus vs. whether yet

CD: numeral, cardinal
mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
fifteen 271,124 dozen quintillion DM2,000 ...

DT: determiner
all an another any both del each either every half la many much nary
neither no some such that the them these this those

EX: existential there
there

IN: preposition or conjunction, subordinating
astride among upon whether out inside pro despite on by throughout
below within for towards near behind atop around if like until below
next into if beside ...

JJ: adjective or numeral, ordinal
third ill-mannered pre-war regrettable oiled calamitous first separable
ectoplasmic battery-powered participatory fourth still-to-be-named
multilingual multi-disciplinary ...

JJR: adjective, comparative
bleaker braver breezier briefer brighter brisker broader bumper busier
calmer cheaper choosier cleaner clearer closer colder commoner costlier
cozier creamier crunchier cuter ...

JJS: adjective, superlative
calmest cheapest choicest classiest cleanest clearest closest commonest
corniest costliest crassest creepiest crudest cutest darkest deadliest
dearest deepest densest dinkiest ...

LS: list item marker
A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
SP-44007 Second Third Three Two * a b c d first five four one six three
two

MD: modal auxiliary
can cannot could couldn't dare may might must need ought shall should
shouldn't will would

NN: noun, common, singular or mass
common-carrier cabbage knuckle-duster Casino afghan shed thermostat
investment slide humour falloff slick wind hyena override subhumanity
machinist ...

NNP: noun, proper, singular
Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
Shannon A.K.C. Meltex Liverpool ...

NNS: noun, common, plural
undergraduates scotches bric-a-brac products bodyguards facets coasts
divestitures storehouses designs clubs fragrances averages
subjectivists apprehensions muses factory-jobs ...

PDT: pre-determiner
all both half many quite such sure this

POS: genitive marker
' 's

PRP: pronoun, personal
hers herself him himself hisself it itself me myself one oneself ours
ourselves ownself self she thee theirs them themselves they thou thy us

PRP$: pronoun, possessive
her his mine my our ours their thy your

RB: adverb
occasionally unabatingly maddeningly adventurously professedly
stirringly prominently technologically magisterially predominately
swiftly fiscally pitilessly ...

RBR: adverb, comparative
further gloomier grander graver greater grimmer harder harsher
healthier heavier higher however larger later leaner lengthier less-
perfectly lesser lonelier longer louder lower more ...

RBS: adverb, superlative
best biggest bluntest earliest farthest first furthest hardest
heartiest highest largest least less most nearest second tightest worst

RP: particle
aboard about across along apart around aside at away back before behind
by crop down ever fast for forth from go high i.e. in into just later
low more off on open out over per pie raising start teeth that through
under unto up up-pp upon whole with you

TO: "to" as preposition or infinitive marker
to

UH: interjection
Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
man baby diddle hush sonuvabitch ...

VB: verb, base form
ask assemble assess assign assume atone attention avoid bake balkanize
bank begin behold believe bend benefit bevel beware bless boil bomb
boost brace break bring broil brush build ...

VBD: verb, past tense
dipped pleaded swiped regummed soaked tidied convened halted registered
cushioned exacted snubbed strode aimed adopted belied figgered
speculated wore appreciated contemplated ...

VBG: verb, present participle or gerund
telegraphing stirring focusing angering judging stalling lactating
hankerin' alleging veering capping approaching traveling besieging
encrypting interrupting erasing wincing ...

VBN: verb, past participle
multihulled dilapidated aerosolized chaired languished panelized used
experimented flourished imitated reunifed factored condensed sheared
unsettled primed dubbed desired ...

VBP: verb, present tense, not 3rd person singular
predominate wrap resort sue twist spill cure lengthen brush terminate
appear tend stray glisten obtain comprise detest tease attract
emphasize mold postpone sever return wag ...

VBZ: verb, present tense, 3rd person singular
bases reconstructs marks mixes displeases seals carps weaves snatches
slumps stretches authorizes smolders pictures emerges stockpiles
seduces fizzes uses bolsters slaps speaks pleads ...

WDT: WH-determiner
that what whatever which whichever

WP: WH-pronoun
that what whatever whatsoever which who whom whosoever

WRB: Wh-adverb
how however whence whenever where whereby whereever wherein whereof why
"""