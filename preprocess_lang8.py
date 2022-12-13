#!/usr/bin/env python
"""
Preprocessing script for Lang-8 v2 data. This script will:
1) Identify English sentence pairs (using langid.py),
2) Remove annotation tags (such as [f-blue], [f-red], and [sline]),
3) Retain only sentence pairs with modifications.
"""

import argparse
import json
# from gcld3 import NNetLanguageIdentifier
import fasttext
import pycountry
import re
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm


# downloaded from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
lid_model=fasttext.load_model("./lid.176.bin")

sline_pattern = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]','[f-red]','[/f-red]','[f-bold]','[/f-bold]']
sent_end = ['.','?','!','"',"'"]


def remove_tags(line):
    for tag in color_tags:
        line = line.replace(tag, '')
    line = sline_pattern.sub('', line)
    line = line.replace('[/sline]', '')
    return re.sub('\s+', ' ', line)


def process(line, language, is_strict=False):
    edited_pairs = set()
    unchanged_pairs = set()
    
    row = json.loads(re.sub(r'[\x00-\x1F]+', '', line))
    extract_lang=pycountry.languages.get(name=language)
    # if (is_strict and row[2] == extract_lang.name) or (not is_strict and extract_lang.name in row[2]):
    if row[2] == extract_lang.name if is_strict else extract_lang.name in row[2]:
        for i in range(len(row[4])):
            src_sent = row[4][i].strip() # remove '"'
            src_sent = re.sub('\s+|\n', ' ', src_sent)
            src_lang = lid_model.predict(src_sent)[0][0].removeprefix('__label__')
            if src_lang != extract_lang.alpha_2:
                continue
            if len(row[5][i]) == 0: # no edits
                unchanged_pairs.add((src_sent, src_sent))
            for tgt_sent in row[5][i]:
                if not tgt_sent:
                    continue
                tgt_sent = tgt_sent.strip()
                tgt_sent = re.sub('\s+|\n', ' ', tgt_sent)
                if tgt_sent == src_sent:
                    unchanged_pairs.add((src_sent, src_sent))
                    continue
                tgt_lang = lid_model.predict(tgt_sent)[0][0].removeprefix('__label__')
                if tgt_lang != extract_lang.alpha_2:
                    continue
                tgt_sent = remove_tags(tgt_sent).strip()
                if not tgt_sent:
                    # if it becomes empty after removing tags
                    continue
                edited_pairs.add((src_sent, tgt_sent))
    return edited_pairs, unchanged_pairs


def parallelize_preprocess(func, iterator, processes):
    """
    Adapted from https://github.com/alvations/sacremoses/blob/master/sacremoses/util.py#L213-L217.
    """
    iterator = tqdm(iterator)
    if processes <= 1:
        return map(func, iterator)
    return Parallel(n_jobs=processes, backend='threading')(delayed(func)(line) for line in iterator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="The path to the data set")
    parser.add_argument("-o", "--output", required=True, help="Edited file prefix")
    parser.add_argument("-k", "--keep", required=True, help="Unchanged file prefix")
    parser.add_argument("-j", "--jobs", type=int, required=False, default=1, help="Number of parallel jobs")
    parser.add_argument("-l", "--language", required=True, help="The language to be extracted")
    parser.add_argument("--strict", action="store_true", help="Strict language identification")
    args = parser.parse_args()

    with open(args.data) as fin, \
         open(args.output, 'w') as fout, \
         open(args.keep, 'w') as fkeep:

        process_func = partial(process, language=args.language, is_strict=args.strict)

        for edited_pairs, unchanged_pairs in parallelize_preprocess(
            process_func, fin.readlines(), args.jobs
        ):
            for src_sent, tgt_sent in edited_pairs:
                print("{}\t{}".format(src_sent, tgt_sent), end='\n', file=fout)
            for src_sent, tgt_sent in unchanged_pairs:
                print("{}\t{}".format(src_sent, tgt_sent), end='\n', file=fkeep)
