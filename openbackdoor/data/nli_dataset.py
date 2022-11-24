"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from .data_processor import DataProcessor

class MnliProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()
        self.path = "./datasets/NLI/mnli"

    def get_examples(self, data_dir, split):
        if data_dir is None:
            data_dir = self.path
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                # example = (text_a+" "+text_b, int(label)-1)
                example = (text_a+" "+text_b, int(label), 0)
                examples.append(example)
                
        return examples

class QnliProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.path = "./datasets/NLI/qnli"
        self.text2lab = {'entailment': '0', 'not_entailment': '1'}
        
    def get_examples(self, data_dir, split):
        if split == 'test':
            split = 'dev'
        if data_dir is None:
            data_dir = self.path
        path = os.path.join(data_dir, "{}.tsv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            _ = f.readline()
            for line in f.readlines():
                idx, context, hypothesis, label = line.strip().split('\t')
                text_a = context
                text_b = hypothesis
                example = (text_a+" "+text_b, int(self.text2lab[label]), 0)
                examples.append(example)

        return examples

class AnliProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.path = "./datasets/NLI/qnli"
        self.text2lab = {'c': '0', 'e': '1', 'n': '2'}
        
    def get_examples(self, data_dir, split):
        if data_dir is None:
            data_dir = self.path
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f.readlines():
                entry = json.loads(line.strip())
                label, context, hypothesis = self.text2lab[entry['label']], entry['context'], entry['hypothesis']
                text_a = context
                text_b = hypothesis
                example = (text_a+" "+text_b, int(label), 0)
                # example = (text_a+" "+text_b, int(label), 0)
                examples.append(example)
                
        return examples

     
PROCESSORS = {
    "mnli" : MnliProcessor,
    "anli" : AnliProcessor,
    "qnli" : QnliProcessor,
}
