from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed)
from datasets import load_dataset
import evaluate
import typer
from rich import print

@dataclass
class TrainConfig:
    model_name: str = 'facebook/bart-base'
    dataset: str = 'ccdv/cnn_dailymail'
    config: str = '3.0.0'
    output_dir: str = 'runs/bart-baseline'
    max_source_length: int = 1024
    max_target_length: int = 128
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    seed: int = 13

def preprocess_function(examples, tokenizer, cfg: TrainConfig):
    model_inputs = tokenizer(examples['article'], max_length=cfg.max_source_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=cfg.max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main(
    output_dir: str = typer.Option('runs/bart-baseline'),
    model_name: str = 'facebook/bart-base',
    dataset: str = 'cnn_dailymail',
    config: str = '3.0.0',
    max_source_length: int = 1024,
    max_target_length: int = 128,
    learning_rate: float = 3e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    seed: int = 13,
):
    cfg = TrainConfig(
        model_name=model_name, dataset=dataset, config=config, output_dir=output_dir,
        max_source_length=max_source_length, max_target_length=max_target_length,
        learning_rate=learning_rate, num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size, per_device_eval_batch_size=per_device_eval_batch_size,
        seed=seed
    )
    set_seed(cfg.seed)
    print(f'[bold]Loading dataset {cfg.dataset} ({cfg.config})[/bold]')
    raw = load_dataset(cfg.dataset, cfg.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenized = raw.map(lambda x: preprocess_function(x, tokenizer, cfg), batched=True, remove_columns=raw['train'].column_names)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge = evaluate.load('rouge')
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy='steps',
        eval_steps=2000,
        save_steps=2000,
        logging_steps=100,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=cfg.num_train_epochs,
        predict_with_generate=True,
        fp16=False
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f'[green]Saved model to {cfg.output_dir}[/green]')

if __name__ == '__main__':
    typer.run(main)
