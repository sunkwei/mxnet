# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import csv

import pickle
import os.path as osp


curr_path = osp.dirname(osp.abspath(__file__))


class LabelUtil:
    _log = None

    # dataPath
    def __init__(self):
        self.load_unicode_set()

    def load_unicode_set(self, vocab_fname=curr_path+'/data/vocab.pkl'):
        self.unicodeFilePath = vocab_fname
        with open(vocab_fname, 'rb') as f:
            d = pickle.load(f)
        self.byChar = d['r_vocab']
        self.byIndex = d['vocab']

    def to_unicode(self, src, index):
        # 1 byte
        code1 = int(ord(src[index + 0]))

        index += 1

        result = code1

        return result, index

    def convert_word_to_grapheme(self, label):

        result = []

        index = 0
        while index < len(label):
            (code, nextIndex) = self.to_unicode(label, index)

            result.append(label[index])

            index = nextIndex

        return result, "".join(result)

    def convert_word_to_num(self, word):

        try:
            label_list, _ = self.convert_word_to_grapheme(word)

            label_num = []

            for char in label_list:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def convert_bi_graphemes_to_num(self, word):
            label_num = []

            for char in word:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)


    def convert_num_to_word(self, num_list):
        try:
            label_list = []
            for num in num_list:
                label_list.append(self.byIndex[num])
            return ''.join(label_list)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def get_count(self):
        try:
            return self.count

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_unicode_file_path(self):
        try:
            return self.unicodeFilePath

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

    def get_blank_index(self):
        return self.byChar["-"]

    def get_space_index(self):
        return self.byChar["$"]



vocab = LabelUtil()
