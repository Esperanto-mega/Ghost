import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels

        return inputs

class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if isinstance(self.tokenizer, LlamaTokenizer):
            # Allow batched inference
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

class Collator4T5(object):
    def init(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        target_texts = [d["labels"] for d in batch]

        features = self.tokenizer(
            text=input_texts,
            text_target=target_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        labels = features["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        features["labels"] = labels

        if "token_type_ids" in features:
            del features["token_type_ids"]

        return features

class ULCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        self.tokenizer.padding_side = "right"

    def __call__(self, batch):

        prompts = [d["input_ids"] for d in batch]
        pos_full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        pos_inputs = self.tokenizer(
            text=pos_full_texts,
            text_target=prompts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        pos_labels = copy.deepcopy(pos_inputs["input_ids"])
        if self.only_train_response:
            pos_labels[pos_labels == self.tokenizer.pad_token_id] = -100
            pos_labels[torch.where(pos_inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        batch_data = {
            "input_ids": pos_inputs["input_ids"],
            "attention_mask": pos_inputs["attention_mask"],
            "labels": pos_labels
        }

        has_negative = any("negative_labels" in d for d in batch)

        if has_negative:
            neg_full_texts = []
            for d in batch:
                if "negative_labels" in d:
                    neg_full_texts.append(d["negative_labels"] + self.tokenizer.eos_token)
                else:
                    neg_full_texts.append(d["input_ids"] + self.tokenizer.eos_token)

            neg_inputs = self.tokenizer(
                text=neg_full_texts,
                text_target=prompts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )

            neg_labels = copy.deepcopy(neg_inputs["input_ids"])
            if self.only_train_response:
                neg_labels[neg_labels == self.tokenizer.pad_token_id] = -100
                neg_labels[torch.where(neg_inputs["labels"] != self.tokenizer.pad_token_id)] = -100

            for i, d in enumerate(batch):
                if "negative_labels" not in d:
                    neg_labels[i] = -100

            batch_data["negative_input_ids"] = neg_inputs["input_ids"]
            batch_data["negative_attention_mask"] = neg_inputs["attention_mask"]
            batch_data["negative_labels"] = neg_labels

        return batch_data