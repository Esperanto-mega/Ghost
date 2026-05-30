import argparse
import json
import os
import sys

import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results

def test_ddp(args):

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    if args.lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        print(f'Load checkpoint from {args.ckpt_path} ......')
        # config = AutoConfig.from_pretrained(args.ckpt_path)
        # config.tie_word_embeddings = False
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            # config=config,
        )
    
    # assert model.config.vocab_size == len(tokenizer)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()


    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)
    
    if local_rank == 0:
        print("Loading head and tail items...")
        index = test_data.indices

        with open(args.head_items, "r") as f:
            head_raw_ids = set(json.load(f))

        with open(args.tail_items, "r") as f:
            tail_raw_ids = set(json.load(f))
            
        print("Calculating GH(G) and Item Popularity (for ARP) from interaction history...")
        inter_data = test_data.inters
        
        total_inter_items = 0
        head_hist_count = 0
        tail_hist_count = 0
        
        head_raw_ids_str = set(str(i) for i in head_raw_ids)
        tail_raw_ids_str = set(str(i) for i in tail_raw_ids)

        seqs = list(inter_data.values())

        item_pop_raw = {}
        
        for seq in seqs:
            hist_seq = seq[:-2]
            for item in hist_seq:
                item_str = str(item)
                
                item_pop_raw[item_str] = item_pop_raw.get(item_str, 0) + 1
                
                if item_str in head_raw_ids_str:
                    head_hist_count += 1
                if item_str in tail_raw_ids_str:
                    tail_hist_count += 1
                total_inter_items += 1
        
        GH_head = head_hist_count / total_inter_items if total_inter_items > 0 else 0.0
        GH_tail = tail_hist_count / total_inter_items if total_inter_items > 0 else 0.0
        print(f"Historical Interactions (except last 2): Total={total_inter_items}")
        print(f"GH(head): {GH_head:.6f}, GH(tail): {GH_tail:.6f}")
        
        def get_token_str(raw_id):
            key = str(raw_id)
            if key in index:
                return "".join(index[key])
            print(f"[Warning]Token not found for id: {raw_id}!!!")
            return None

        head_ids = set()
        for id in head_raw_ids:
            t_str = get_token_str(id)
            if t_str: 
                head_ids.add(t_str)
        tail_ids = set()
        for id in tail_raw_ids:
            t_str = get_token_str(id)
            if t_str: 
                tail_ids.add(t_str)
                
        token_popularity = {}
        for raw_id, pop in item_pop_raw.items():
            t_str = get_token_str(raw_id)
            if t_str:
                token_popularity[t_str] = pop

        print(f"Head IDs Intersect Tail IDs: {len(head_ids & tail_ids)}")
        print(f"Head Items: {len(head_raw_ids)}")
        print(f"Tail Items: {len(tail_raw_ids)}")

    if local_rank == 0:
        print("data num:", len(test_data))

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results =[]
    with torch.no_grad():

        for prompt_id in prompt_ids:
            head_tail_counts = {
                "head@1": 0,
                "head@5": 0,
                "head@10": 0,
                "tail@1": 0,
                "tail@5": 0,
                "tail@10": 0
            }

            group_item_counts = {
                "head@1": 0, 
                "head@5": 0, 
                "head@10": 0,
                "tail@1": 0, 
                "tail@5": 0, 
                "tail@10": 0
            }
            
            arp_sums = {
                "arp@1": 0.0,
                "arp@5": 0.0,
                "arp@10": 0.0
            }

            if local_rank == 0:
                print("Start prompt: ",prompt_id)

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            head_metrics_results = {}
            tail_metrics_results = {}
            
            total = 0
            head_total = 0
            tail_total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1] # 'targets' is semantic id
                # print('target:',targets)
                bs = len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        output = model.module.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            # prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            length_penalty=args.length_penalty,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams -1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output, scores, targets, num_beams,
                                            all_items=all_items if args.filter_items else None)

                bs_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                total += sum(bs_gather_list)
                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

                target_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=targets, object_list=target_gather_list)

                if local_rank == 0:
                    all_device_topk_res =[]
                    for ga_res in res_gather_list:
                        all_device_topk_res += ga_res

                    all_device_targets = []
                    for ga_targets in target_gather_list:
                        all_device_targets += ga_targets
                    
                    batch_hits_data = []
                    head_hits_data = []
                    tail_hits_data = []

                    for res, target in zip(all_device_topk_res, all_device_targets):
                        batch_hits_data.append(res["hits"])
                        if target in head_ids:
                            head_hits_data.append(res["hits"])
                        # elif target in tail_ids:
                        else:
                            tail_hits_data.append(res["hits"])
                    
                    batch_metrics_res = get_metrics_results(batch_hits_data, metrics)
                    for m, res in batch_metrics_res.items():
                        metrics_results[m] = metrics_results.get(m, 0) + res

                    head_total += len(head_hits_data)
                    if len(head_hits_data) > 0:
                        head_batch_metrics = get_metrics_results(head_hits_data, metrics)
                        for m, res in head_batch_metrics.items():
                            head_metrics_results[m] = head_metrics_results.get(m, 0) + res

                    tail_total += len(tail_hits_data)
                    if len(tail_hits_data) > 0:
                        tail_batch_metrics = get_metrics_results(tail_hits_data, metrics)
                        for m, res in tail_batch_metrics.items():
                            tail_metrics_results[m] = tail_metrics_results.get(m, 0) + res

                    if (step + 1) % 100 == 0:
                        temp = {m: metrics_results[m] / total for m in metrics_results}
                        print("Overall:", temp)
                        temp = {m: head_metrics_results[m] / head_total for m in head_metrics_results}
                        print("Head Items:", temp)
                        temp = {m: tail_metrics_results[m] / tail_total for m in tail_metrics_results}
                        print("Tail Items:", temp)
                    
                    for res in all_device_topk_res:
                        top_items = res["items"]

                        top1 = top_items[:1]
                        top5 = top_items[:5]
                        top10 = top_items[:10]

                        if any(i in head_ids for i in top1):
                            head_tail_counts["head@1"] += 1
                        if any(i in head_ids for i in top5):
                            head_tail_counts["head@5"] += 1
                        if any(i in head_ids for i in top10):
                            head_tail_counts["head@10"] += 1

                        if any(i in tail_ids for i in top1):
                            head_tail_counts["tail@1"] += 1
                        if any(i in tail_ids for i in top5):
                            head_tail_counts["tail@5"] += 1
                        if any(i in tail_ids for i in top10):
                            head_tail_counts["tail@10"] += 1

                        group_item_counts["head@1"] += sum([1 for i in top1 if i in head_ids])
                        group_item_counts["head@5"] += sum([1 for i in top5 if i in head_ids])
                        group_item_counts["head@10"] += sum([1 for i in top10 if i in head_ids])

                        group_item_counts["tail@1"] += sum([1 for i in top1 if i in tail_ids])
                        group_item_counts["tail@5"] += sum([1 for i in top5 if i in tail_ids])
                        group_item_counts["tail@10"] += sum([1 for i in top10 if i in tail_ids])

                        arp_sums["arp@1"] += sum([token_popularity.get(i, 0) for i in top1]) / 1.0
                        arp_sums["arp@5"] += sum([token_popularity.get(i, 0) for i in top5]) / 5.0
                        arp_sums["arp@10"] += sum([token_popularity.get(i, 0) for i in top10]) / 10.0

                dist.barrier()

            if local_rank == 0:
                for m in list(metrics_results.keys()):
                    metrics_results[m] = metrics_results[m] / total

                for m in list(head_metrics_results.keys()):
                    metrics_results[f"Head_{m}"] = head_metrics_results[m] / head_total if head_total > 0 else 0.0
                
                for m in list(tail_metrics_results.keys()):
                    metrics_results[f"Tail_{m}"] = tail_metrics_results[m] / tail_total if tail_total > 0 else 0.0
                
                print(f"Dataset Target Distribution -> Head Targets: {head_total}, Tail Targets: {tail_total}")

                for k in[1, 5, 10]:
                    GP_head = group_item_counts[f"head@{k}"] / (total * k)
                    GP_tail = group_item_counts[f"tail@{k}"] / (total * k)
                    
                    GU_head = GP_head - GH_head
                    GU_tail = GP_tail - GH_tail
                    
                    metrics_results[f"GU_head@{k}"] = GU_head
                    metrics_results[f"GU_tail@{k}"] = GU_tail
                    metrics_results[f'MGU@{k}'] = 0.5 * (abs(GU_head) + abs(GU_tail))
                    metrics_results[f"ARP@{k}"] = arp_sums[f"arp@{k}"] / total
                    
                for k, v in head_tail_counts.items():
                    metrics_results[k] = f'{v}/{total}={v/total:.6f}'

                all_prompt_results.append(metrics_results)
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")

            dist.barrier()

    dist.barrier()

    if local_rank == 0:
        mean_results = {}
        min_results = {}
        max_results = {}

        all_metrics_to_summary = metrics + [f"Head_{m}" for m in metrics] + [f"Tail_{m}" for m in metrics] + [f"GU_head@{k}" for k in [1, 5, 10]] + [f"GU_tail@{k}" for k in [1, 5, 10]] + [f"MGU@{k}" for k in [1, 5, 10]] + [f"ARP@{k}" for k in [1, 5, 10]]
        
        for m in all_metrics_to_summary:
            if m in all_prompt_results[0]:
                all_res = [_[m] for _ in all_prompt_results]
                mean_results[m] = sum(all_res)/len(all_res)
                min_results[m] = min(all_res)
                max_results[m] = max(all_res)

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

        save_data={}
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print("Save file: ", args.results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)   

    args = parser.parse_args()

    test_ddp(args)