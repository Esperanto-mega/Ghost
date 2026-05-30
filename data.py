import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
from prompt import sft_prompt, all_prompt
import numpy as np


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix
        
        self.inter_file = args.inter_file

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer, is_t5 = False):

        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    # token_id = tokenizer(token)["input_ids"][1]
                    ids = tokenizer(token)["input_ids"]
                    if len(ids) > 1:
                        if is_t5:
                            token_id = ids[0]
                        else:
                            token_id = ids[-1]
                    else:
                        continue
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
            
        # print(self.allowed_tokens)
            
        if is_t5:
            sep = tokenizer("Response:")["input_ids"][:-1]
        else:
            sep = tokenizer("Response:")["input_ids"][1:]
            
        print('sep:', sep)

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            # print('sentence:', sentence)
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError

class SeqRecDataset(BaseDataset):
        
    def __init__(
        self, args, mode="train",
        prompt_sample_num=1, prompt_id=0, sample_num=-1, 
    ): 
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.prompts = all_prompt["seqrec"]
        
        if self.mode == 'train':
            if os.path.exists(args.head_items):
                with open(args.head_items, 'r') as f:
                    self.head_items = set(json.load(f))
            else:
                self.head_items = set()
            print(f"Loaded {len(self.head_items)} head items.")

        if self.mode == 'train':
            self.embedding_file = args.embed_file
            self.negative_k = args.negative_k
            self.hard_negative_map = {} 

        # load data
        self._load_data()
        self._remap_items()
        
        if self.mode == 'train' and self.embedding_file:
            # self._init_hard_negative_pool()
            self._init_hard_negative_pool_v2()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):

        with open(self.inter_file, 'r') as f:
            self.inters = json.load(f)
        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _init_hard_negative_pool(self):

        print(f"Loading embeddings from {self.embedding_file}...")

        item_embeddings = torch.load(self.embedding_file, map_location='cpu')
        

        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
        
        all_item_ids = list(self.indices.keys()) # string IDs
        

        head_indices = []
        tail_indices = []
        
        id_map = {} 
        
        for iid in all_item_ids:
            try:
                idx = int(iid)
                id_map[iid] = idx
                if int(iid) in self.cold_items:
                    tail_indices.append(idx)
                else:
                    head_indices.append(idx)
            except ValueError:
                continue

        head_indices_tensor = torch.tensor(head_indices).long()
        tail_indices_tensor = torch.tensor(tail_indices).long()
        
        head_embeds = item_embeddings[head_indices_tensor] # [Num_Head, Dim]
        tail_embeds = item_embeddings[tail_indices_tensor] # [Num_Tail, Dim]
        
        print(f"Computing hard negatives for Head items (Pool: {len(head_indices)}, Target: {len(tail_indices)})...")

        batch_size = 1024
        for i in tqdm(range(0, len(head_indices), batch_size), desc="Head->Tail"):
            batch_head_idxs = head_indices_tensor[i : i + batch_size]
            batch_embeds = item_embeddings[batch_head_idxs]
            
            # Similarity: [Batch, Num_Tail]
            sim_scores = torch.matmul(batch_embeds, tail_embeds.t())
            
            # Top-K
            _, topk_indices = torch.topk(sim_scores, k=self.negative_k, dim=1)
            
            for j, original_tensor_idx in enumerate(batch_head_idxs.tolist()):
                neg_tensor_idxs = tail_indices_tensor[topk_indices[j]].tolist()
                self.hard_negative_map[str(original_tensor_idx)] = [str(x) for x in neg_tensor_idxs]

        print(f"Computing hard negatives for Tail items (Pool: {len(tail_indices)}, Target: {len(head_indices)})...")
        for i in tqdm(range(0, len(tail_indices), batch_size), desc="Tail->Head"):
            batch_tail_idxs = tail_indices_tensor[i : i + batch_size]
            batch_embeds = item_embeddings[batch_tail_idxs]
            
            # Similarity: [Batch, Num_Head]
            sim_scores = torch.matmul(batch_embeds, head_embeds.t())
            
            # Top-K
            _, topk_indices = torch.topk(sim_scores, k=self.negative_k, dim=1)
            
            for j, original_tensor_idx in enumerate(batch_tail_idxs.tolist()):
                neg_tensor_idxs = head_indices_tensor[topk_indices[j]].tolist()
                self.hard_negative_map[str(original_tensor_idx)] = [str(x) for x in neg_tensor_idxs]

    def _init_hard_negative_pool_v2(self, candidate_pool_size=50):

        print(f"Loading embeddings from {self.embedding_file}...")
        item_embeddings = torch.load(self.embedding_file, map_location='cpu')
        item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
        
        all_item_ids = list(self.indices.keys()) # string IDs
        
        head_indices = []
        tail_indices = []
        
        idx_to_str = {} 
        
        for iid in all_item_ids:
            try:
                idx = int(iid)
                idx_to_str[idx] = iid
                if int(iid) in self.head_items:
                    head_indices.append(idx)
                else:
                    tail_indices.append(idx)

            except ValueError:
                continue

        head_indices_tensor = torch.tensor(head_indices).long()
        tail_indices_tensor = torch.tensor(tail_indices).long()
        
        head_embeds = item_embeddings[head_indices_tensor] # [Num_Head, Dim]
        tail_embeds = item_embeddings[tail_indices_tensor] # [Num_Tail, Dim]
        
        def get_semantic_similarity(id1_str, id2_str):
            seq1 = self.indices[id1_str]
            seq2 = self.indices[id2_str]
            
            rq_seq1 = seq1[:4]
            rq_seq2 = seq2[:6]
            # rq_seq1 = seq1[:-1]
            # rq_seq2 = seq2[:-1]

            match_len = 0
            for t1, t2 in zip(rq_seq1, rq_seq2):
                if t1 == t2:
                    match_len += 1
                else:
                    break
                    
            return match_len
        
        print(f"Computing hard negatives V2 for Head items (Pool: {len(head_indices)}, Target: {len(tail_indices)})...")
        batch_size = 1024
        for i in tqdm(range(0, len(head_indices), batch_size), desc="Head->Tail"):
            batch_head_idxs = head_indices_tensor[i : i + batch_size]
            batch_embeds = item_embeddings[batch_head_idxs]

            sim_scores = torch.matmul(batch_embeds, tail_embeds.t())

            M = min(candidate_pool_size, sim_scores.shape[1])
            _, topm_indices = torch.topk(sim_scores, k=M, dim=1)
            
            for j, original_tensor_idx in enumerate(batch_head_idxs.tolist()):
                target_str_id = idx_to_str[original_tensor_idx]
                neg_tensor_idxs = tail_indices_tensor[topm_indices[j]].tolist()

                candidates_sem_scores = []
                for neg_idx in neg_tensor_idxs:
                    neg_str_id = idx_to_str[neg_idx]
                    sem_sim = get_semantic_similarity(target_str_id, neg_str_id)
                    candidates_sem_scores.append((sem_sim, neg_str_id))

                candidates_sem_scores.sort(key=lambda x: x[0])

                selected_negatives = [item[1] for item in candidates_sem_scores[:self.negative_k]]
                self.hard_negative_map[str(original_tensor_idx)] = selected_negatives

        print(f"Computing hard negatives V2 for Tail items (Pool: {len(tail_indices)}, Target: {len(head_indices)})...")
        for i in tqdm(range(0, len(tail_indices), batch_size), desc="Tail->Head"):
            batch_tail_idxs = tail_indices_tensor[i : i + batch_size]
            batch_embeds = item_embeddings[batch_tail_idxs]
            
            sim_scores = torch.matmul(batch_embeds, head_embeds.t())
            
            M = min(candidate_pool_size, sim_scores.shape[1])
            _, topm_indices = torch.topk(sim_scores, k=M, dim=1)
            
            for j, original_tensor_idx in enumerate(batch_tail_idxs.tolist()):
                target_str_id = idx_to_str[original_tensor_idx]
                neg_tensor_idxs = head_indices_tensor[topm_indices[j]].tolist()
                
                candidates_sem_scores = []
                for neg_idx in neg_tensor_idxs:
                    neg_str_id = idx_to_str[neg_idx]
                    sem_sim = get_semantic_similarity(target_str_id, neg_str_id)
                    candidates_sem_scores.append((sem_sim, neg_str_id))
                    
                candidates_sem_scores.sort(key=lambda x: x[0])
                
                selected_negatives = [item[1] for item in candidates_sem_scores[:self.negative_k]]
                self.hard_negative_map[str(original_tensor_idx)] = selected_negatives
    
    def _get_negative_item(self, target_item_id):

        if self.embedding_file and target_item_id in self.hard_negative_map:
            candidates = self.hard_negative_map[target_item_id]
            neg_item_id = random.choice(candidates)
            return "".join(self.indices[neg_item_id])

        self._init_hard_negative_pool_v2()
        # self._init_hard_negative_pool()
        # self._init_negative_pool()
        while True:
            neg_item_id = random.choice(self.all_item_ids)

            if neg_item_id != target_item_id:
                return "".join(self.indices[neg_item_id])

    def _process_train_data(self):
        inter_data = []
        
        for uid in self.remapped_inters:
            token_items = self.remapped_inters[uid][:-2]
            original_items = self.inters[uid][:-2]
            
            for i in range(1, len(token_items)):
                one_data = dict()
                one_data["item"] = token_items[i]
                one_data["item_id"] = str(original_items[i])
                
                history_tokens = token_items[:i]
                if self.max_his_len > 0:
                    history_tokens = history_tokens[-self.max_his_len:]
                if self.add_prefix:
                    history_tokens = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history_tokens)]
                one_data["inters"] = self.his_sep.join(history_tokens)
                inter_data.append(one_data)
                

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output
    
    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input_ids, labels = self._get_text_data(d, prompt)

        if self.mode == 'train':
            d_neg = copy.deepcopy(d)

            target_id = d.get('item_id', None)
            if target_id is None: 
                 pass 
                 
            d_neg['item'] = self._get_negative_item(target_id)
            
            _, neg_labels = self._get_text_data(d_neg, prompt)
            
            return dict(input_ids=input_ids, labels=labels, negative_labels=neg_labels)
        
        return dict(input_ids=input_ids, labels=labels)

class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["fusionseqrec"]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError


    def _load_data(self):

        with open(self.inter_file, 'r') as f:
            self.inters = json.load(f)
        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = "".join(self.indices[str(items[i])])
                one_data["title"] = self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                inters = ["".join(self.indices[str(j)]) for j in history]
                inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]


                if self.add_prefix:
                    inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                    inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

                one_data["inters"] = self.his_sep.join(inters)
                one_data["inter_titles"] = self.his_sep.join(inter_titles)
                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-2])])
            one_data["title"] = self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-2])]["description"]


            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = "".join(self.indices[str(items[-1])])
            one_data["title"] = self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            inters = ["".join(self.indices[str(j)]) for j in history]
            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]

            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]
                inter_titles = [str(k + 1) + ". " + item_title for k, item_title in enumerate(inter_titles)]

            one_data["inters"] = self.his_sep.join(inters)
            one_data["inter_titles"] = self.his_sep.join(inter_titles)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)


        return dict(input_ids=input, labels=output)


class ItemFeatDataset(BaseDataset):

    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()
        self.feat_data = self._process_data()



    def _load_data(self):

        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)


    def _process_data(self):

        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            index = "".join(self.indices[iid])
            feat["item"] = index
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data


    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)


class ItemSearchDataset(BaseDataset):

    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["itemsearch"]

        # load data
        self._load_data()
        self.search_data = self._process_data()



    def _load_data(self):

        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)


    def _process_data(self):

        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == 'train':
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == 'test':
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            index = "".join(self.indices[str(iid)])
            one_data["item"] = index

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            inters = ["".join(self.indices[str(i)]) for i in inters]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["inters"] = self.his_sep.join(inters)

            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == 'test':
            return len(self.search_data)
        else:
            return len(self.search_data)


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)



class PreferenceObtainDataset(BaseDataset):

    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # load data
        self._load_data()
        self._remap_items()

        self.preference_data = self._process_data()



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)
        with open(self.inter_file, 'r') as f:
            self.inters = json.load(f)
        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_data(self):

        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            inters = self.remapped_inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            if self.add_prefix:
                inters = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(inters)]

            one_data["explicit_preferences"] = user_ep
            one_data["inters"] = self.his_sep.join(inters)

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)

            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num


    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)





class SeqRecTestDataset(BaseDataset):

    def __init__(self, args, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompt = all_prompt["seqrec"][self.prompt_id]

        # load data
        self._load_data()
        self._remap_items()

        self.inter_data = self._process_test_data()

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)

            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

        self.prompt = all_prompt["seqrec"][self.prompt_id]

    def __len__(self):

        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")

        return input, response

    def __getitem__(self, index):

        d = self.inter_data[index]
        input, target = self._get_text_data(d, self.prompt)

        return dict(input_ids=input, labels=target)
    








# class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1, slide_window = False, slide_size = 1,
                 cold_items = '/nfsdata/huopeng/Mar0304/LC-Rec/data/Instruments_test/Instruments_total_ID_list-cold.json'):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]
        
        self.slide_window = slide_window
        self.slide_inters, self.cold_items = None, None
        if self.slide_window:
            self.slide_size = slide_size
            with open(cold_items, 'r') as f:
                self.cold_items = set(json.load(f))
            self.slide_inters = {}

        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(self.inter_file, 'r') as f:
            self.inters = json.load(f)
        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)

        if self.slide_window:
            num_inters = len(self.slide_inters)
            for inter in self.inters.values():
                inter = inter[:-2]
                for i in range(1, len(inter)):
                    if inter[i] in self.cold_items:
                        for j in range(0, i - 1):
                            self.slide_inters[str(num_inters)] = inter[j:i+1]
                            num_inters += 1
                            
            assert num_inters == len(self.slide_inters)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items
        
        if self.slide_window:
            self.remapped_slide_inters = dict()
            for uid, items in self.slide_inters.items():
                new_items = ["".join(self.indices[str(i)]) for i in items]
                self.remapped_slide_inters[uid] = new_items

    def _init_negative_pool(self):
        if not hasattr(self, 'all_item_ids'):
            self.all_item_ids = list(self.indices.keys())

    def _get_negative_item(self, target_item_idx):
        self._init_negative_pool()
        while True:
            neg_item_id = random.choice(self.all_item_ids)
            neg_item_token = "".join(self.indices[neg_item_id])
            
            if neg_item_token != target_item_idx:
                return neg_item_token

    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)
                
        if self.slide_window:
            for inter in self.remapped_slide_inters.values():
                one_data = dict()
                one_data['item'] = inter[-1]
                history = inter[:-1]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = self.his_sep.join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input, output = self._get_text_data(d, prompt)
                    self.valid_text_data.append({"input_ids": input, "labels": output})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input, output = self._get_text_data(d, prompt)
                self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output
    
    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        input_ids, labels = self._get_text_data(d, prompt)

        if self.mode == 'train':
            d_neg = copy.deepcopy(d)
            d_neg['item'] = self._get_negative_item(d['item'])

            _, neg_labels = self._get_text_data(d_neg, prompt)
            
            return dict(input_ids=input_ids, labels=labels, negative_labels=neg_labels)
        
        return dict(input_ids=input_ids, labels=labels)
