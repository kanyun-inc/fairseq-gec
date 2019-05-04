# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os

from fairseq.tokenizer import tokenize_line


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1, copy_ext_dict=False, copy_src_words=None):
        nseq, ntok = 0, 0
        replaced = Counter()
        copied = Counter()

        def replaced_consumer(word, idx):
            if (idx == dict.unk_index or idx >= len(dict)) and word != dict.unk_word:
                replaced.update([word])
            if idx >= len(dict) and copy_src_words is not None:
                copied.update([word])

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                words = []
                ids = dict.encode_line(  # todo: change all encode_line
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                        copy_ext_dict=copy_ext_dict,
                        copy_src_words=None if copy_src_words is None else copy_src_words[nseq],
                        out_words=words,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids, words)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced, 'copied': copied}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
