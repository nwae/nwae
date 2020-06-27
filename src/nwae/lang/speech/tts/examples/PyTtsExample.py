# -*- coding: utf-8 -*-

import pyttsx3
import random

#  In Mac OS，there are all chinese voice
voices_zh_cn = [
    'com.apple.speech.synthesis.voice.mei-jia',
    # 'com.apple.speech.synthesis.voice.sin-ji.premium',
    'com.apple.speech.synthesis.voice.ting-ting'
]

engine = pyttsx3.init() # object creation

rate = engine.getProperty('rate')   # getting details of current speaking rate
print ('Speech rate = ' + str(rate))                        #printing current voice rate
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print ('Speech volume = ' + str(volume))                          #printing current volume level

voice_random = random.choice(voices_zh_cn)
print('Using voice "' + str(voice_random) + '"')
engine.setProperty('voice', voice_random)

""" BASIC SPEECH """
# engine.say('Good morning')
# engine.say(u'Привет как дела?')
#engine.say('너무 아프다')
engine.say(u'我爱你')
engine.say('你好')
engine.say('我的说率是 ' + str(rate))
engine.runAndWait()
engine.stop()
exit(0)


""" RATE"""
engine.setProperty('rate', 125)     # setting up new voice rate

"""VOLUME"""
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
# voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
#engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

engine.say('你好')
engine.say('我的说率是 ' + str(rate))
engine.runAndWait()
engine.stop()