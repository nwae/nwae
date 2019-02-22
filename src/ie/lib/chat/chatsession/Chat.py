# -*- coding: utf-8 -*-

import os
import pandas as pd
import threading


class Chat:

    CHAT_HISTORY_COL_DATETIME = 'DateTime'
    CHAT_HISTORY_COL_CHATID = 'ChatID'
    CHAT_HISTORY_COL_SPEAKER = 'Speaker'
    CHAT_HISTORY_COL_SPEAKER_NAME = 'SpeakerName'
    CHAT_HISTORY_COL_MESSAGE = 'Message'

    def __init__(self,
                 chatid,
                 participants,
                 participant_ids):
        # Unique ID to identify this chat
        self.chatid = chatid
        # List of clients participating in this chat
        self.participants = participants
        # List of client IDs participating in this chat
        self.participant_ids = participant_ids

        #
        # General containers
        #
        # For all clients, chat history
        self.chat_history = pd.DataFrame({
            Chat.CHAT_HISTORY_COL_CHATID: [],
            Chat.CHAT_HISTORY_COL_DATETIME: [],
            Chat.CHAT_HISTORY_COL_SPEAKER: [],
            Chat.CHAT_HISTORY_COL_SPEAKER_NAME: [],
            Chat.CHAT_HISTORY_COL_MESSAGE: []
        })

        self.mutex = threading.Lock()
        self.started = False
        return

    def new_message(self, datetime, speaker, speaker_name, msg):
        self.mutex.acquire()

        newline = pd.DataFrame({
            Chat.CHAT_HISTORY_COL_CHATID: [self.chatid],
            Chat.CHAT_HISTORY_COL_DATETIME: [datetime],
            Chat.CHAT_HISTORY_COL_SPEAKER: [speaker],
            Chat.CHAT_HISTORY_COL_SPEAKER_NAME: [speaker_name],
            Chat.CHAT_HISTORY_COL_MESSAGE: [msg]
        })
        self.chat_history = self.chat_history.append(newline)
        # No need actually since this data frame only handles a single chat id
        self.chat_history = self.chat_history.reset_index(drop=True)

        self.mutex.release()

    def get_chat_id(self):
        return self.chatid

    def get_chat_history(self):
        return self.chat_history.copy()

    def get_participants(self):
        return self.participants

    def get_participant_ids(self):
        return self.participant_ids

    def start_chat(self):
        self.started = True

    def is_started(self):
        return self.started

    def flush_chatlog_to_csv(self, filepath_chatlog):
        self.mutex.acquire()

        if os.path.isfile(filepath_chatlog):
            # File exists, don't write header, just append
            self.chat_history.to_csv(path_or_buf=filepath_chatlog, mode='a', header=False)
        else:
            # File don't exist, write new file with header
            self.chat_history.to_csv(path_or_buf=filepath_chatlog, mode='w', header=True)

        self.mutex.release()
