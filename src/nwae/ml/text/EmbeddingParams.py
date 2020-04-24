# -*- coding: utf-8 -*-


class EmbeddingParams:

    def __init__(
            self,
            x              = None,
            x_original     = None,
            y              = None,
            y_original     = None,
            x_one_hot_dict = None,
            y_one_hot_dict = None,
            max_sent_len   = None,
            max_label_val  = None,
            vocab_size     = None
    ):
        self.x = x
        self.x_original = x_original
        self.y = y
        self.y_original = y_original
        self.x_one_hot_dict = x_one_hot_dict
        self.y_one_hot_dict = y_one_hot_dict
        self.max_sent_len = max_sent_len
        self.max_label_val = max_label_val
        self.vocab_size = vocab_size

