import argparse
import os
import sys
from typing import List
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils import *
from collator import Collator, ULCollator
from transformers import Trainer

import torch.nn.functional as F

class ULTrainer(Trainer):
    def __init__(self, ul_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ul_alpha = ul_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos_input_ids = inputs["input_ids"]
        pos_attention_mask = inputs["attention_mask"]
        pos_labels = inputs["labels"]

        if "negative_input_ids" not in inputs:
            outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask, labels=pos_labels)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        neg_input_ids = inputs["negative_input_ids"]
        neg_attention_mask = inputs["negative_attention_mask"]
        neg_labels = inputs["negative_labels"]

        max_len = max(pos_input_ids.shape[1], neg_input_ids.shape[1])
        # pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0

        def pad_to_max(tensor, pad_value):
            if tensor.shape[1] < max_len:
                return F.pad(tensor, (0, max_len - tensor.shape[1]), value=pad_value)
            return tensor

        p_ids = pad_to_max(pos_input_ids, pad_token_id)
        p_mask = pad_to_max(pos_attention_mask, 0)
        p_labels = pad_to_max(pos_labels, -100)

        n_ids = pad_to_max(neg_input_ids, pad_token_id)
        n_mask = pad_to_max(neg_attention_mask, 0)
        n_labels_dummy = torch.full((n_ids.shape[0], max_len), -100, dtype=p_labels.dtype, device=p_labels.device)

        all_input_ids = torch.cat([p_ids, n_ids], dim=0)
        all_attention_mask = torch.cat([p_mask, n_mask], dim=0)
        all_labels = torch.cat([p_labels, n_labels_dummy], dim=0)

        all_outputs = model(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            labels=all_labels
        )

        mle_loss = all_outputs.loss

        batch_size = pos_input_ids.shape[0]
        neg_logits = all_outputs.logits[batch_size:]

        neg_labels_padded = pad_to_max(neg_labels, -100)

        shift_logits = neg_logits[..., :-1, :].contiguous()
        shift_labels = neg_labels_padded[..., 1:].contiguous()

        mask = shift_labels.ne(-100).float()
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        probs = torch.exp(target_log_probs)
        ul_loss_per_token = -torch.log(torch.clamp(1.0 - probs, min=1e-5))

        ul_loss = (ul_loss_per_token * mask).sum() / (mask.sum() + 1e-9)

        total_loss = mle_loss + self.ul_alpha * ul_loss

        # print('UL Loss: {:.4f}, MLE Loss: {:.4f}, Total Loss: {:.4f}'.format(ul_loss.item(), mle_loss.item(), total_loss.item()))

        return (total_loss, all_outputs) if return_outputs else total_loss


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}

    # config = LlamaConfig.from_pretrained(args.base_model)
    config = AutoConfig.from_pretrained(args.base_model)
    # tokenizer = LlamaTokenizer.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0
    gradient_checkpointing = True

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = ULCollator(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        # dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    ul_alpha = getattr(args, 'ul_alpha', 1.0)

    trainer = ULTrainer(
        ul_alpha=ul_alpha,
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=gradient_checkpointing,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=50,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
            remove_unused_columns=False,
        ),
        processing_class=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
