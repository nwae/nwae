# -*- coding: utf-8 -*-

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
# For commercial use where you can buy more API calls,
# use https://cloud.google.com/speech-to-text/ instead.
# This is only for personal use with usage limits that can't
# be increased
#  > pip install SpeechRecognition
import speech_recognition as sr
# To get audio from microphone
#  > brew install portaudio
#  > pip install PyAudio
from nwae.utils.audio.AudioUtils import AudioUtils
import re
from nwae.lang.LangFeatures import LangFeatures


class SpeechRecognition:

    """
    Caution: The default key provided by SpeechRecognition is for testing purposes only, and Google may revoke it
    at any time. It is not a good idea to use the Google Web Speech API in production. Even with a valid API key,
    you’ll be limited to only 50 requests per day, and there is no way to raise this quota. Fortunately,
    SpeechRecognition’s interface is nearly identical for each API, so what you learn today will be easy to translate
    to a real-world project.
    """
    ENGINE_GOOGLE = 'google'
    # Need account
    # https://cloud.google.com/speech-to-text/docs/quickstart
    ENGINE_GOOGLE_CLOUD = 'google_cloud'
    # Need account
    # https://azure.microsoft.com/en-ca/pricing/details/cognitive-services/speech-api/
    ENGINE_BING = 'bing'

    SOURCE_MIC = 'microphone'
    SOURCE_FILE = 'file'

    def __init__(
            self,
            lang = LangFeatures.LANG_KO,
            audio_source = SOURCE_MIC,
            audio_file = None,
            engine = ENGINE_GOOGLE,
            auth_info = None
    ):
        self.lang = lang
        self.audio_source = audio_source
        self.audio_file = audio_file
        self.engine = engine
        self.auth_info = auth_info

        Log.important(
            str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
            + ': Language "' + str(self.lang) + '" audio source "' + str(self.audio_source) + '"'
        )

        if self.audio_source == self.SOURCE_MIC:
            self.sr_source = sr.Microphone()
        elif self.audio_source == self.SOURCE_FILE:
            self.sr_source = sr.AudioFile(audio_file)
        else:
            raise Exception(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Unsupported audio source "' + str(self.audio_source) + '"'
            )
        return

    def recognize(self):
        if self.audio_source == self.SOURCE_MIC:
            return self.__recognize_mic()
        elif self.audio_source == self.SOURCE_FILE:
            return self.__recognize_file()

    def __recognize_mic(self):
        # Initialize recognizer class (for recognizing the speech)
        r = sr.Recognizer()

        with sr.Microphone() as source:
            print('Start talking')
            audio_text = r.listen(source)
            print('Done')
            try:
                # using google speech recognition
                text = r.recognize_google(audio_text, language=self.lang)
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Recognized "' + str(self.lang) + '" text "' + str(text)
                    + '" from mic "' + str(self.audio_file) + '"'
                )
                return text
            except Exception as ex:
                Log.error(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Exception: ' + str(ex)
                )

    def __recognize_file(self):
        need_convert_format = re.sub(
            pattern = '(.*[.])([a-zA-Z0-9]+$)',
            repl    = '\\2',
            string  = self.audio_file
        ).lower() != 'wav'
        audio_filepath_wav = self.audio_file

        if need_convert_format:
            Log.important(
                str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                + ': Converting "' + str(self.audio_file) + '" to wav format..'
            )
            audio_filepath_wav = AudioUtils().convert_format(
                filepath = self.audio_file
            )

        # Initialize recognizer class (for recognizing the speech)
        r = sr.Recognizer()

        # Reading Audio file as source
        # listening the audio file and store in audio_text variable

        with sr.AudioFile(audio_filepath_wav) as source:
            audio_text = r.listen(source)

            # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
            try:

                if self.engine == SpeechRecognition.ENGINE_GOOGLE:
                    text = r.recognize_google(audio_text, language=self.lang)
                elif self.engine == SpeechRecognition.ENGINE_GOOGLE_CLOUD:
                    text = r.recognize_google_cloud(audio_text, credentials_json=self.auth_info, language=self.lang)
                elif self.engine == SpeechRecognition.ENGINE_BING:
                    text = r.recognize_bing(audio_text, key=self.auth_info, language=self.lang)
                else:
                    raise Exception(
                        str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                        + ': Unsuported engine "' + str(self.engine) + '".'
                    )
                Log.info(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Converting audio transcripts into text ...'
                )
                Log.important(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Recognized "' + str(self.lang) + '" text "' + str(text)
                    + '" from audio file "' + str(self.audio_file) + '"'
                )
                return text
            except Exception as ex:
                Log.error(
                    str(self.__class__) + ' ' + str(getframeinfo(currentframe()).lineno)
                    + ': Exception converting audio transcript from "' + str(self.audio_file)
                    + '": ' + str(ex)
                )


if __name__ == '__main__':
    audio_filepath_mp3 = '/usr/local/git/nwae/nwae.lang/app.data/voice-recordings/hi.m4a'
    audio_filepath_wav = AudioUtils().convert_format(
        filepath = audio_filepath_mp3
    )

    print('Playing audio from "' + str(audio_filepath_wav) + '"')
    AudioUtils().play_wav(
        wav_filepath = audio_filepath_wav,
        play_secs = 2
    )

    text = SpeechRecognition(
        lang = LangFeatures.LANG_RU_RU,
        audio_source = SpeechRecognition.SOURCE_FILE,
        audio_file   = audio_filepath_wav
    ).recognize()
    print('***** ', text)

    text = SpeechRecognition(
        lang = LangFeatures.LANG_KO
    ).recognize()
    print('*****', text)

    exit(0)
