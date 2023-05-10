from modeling_royal_canine import LlamaWithCanineForCausalLM, ByT5Tokenizer, LlamaWithCanineConfig

from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_int8_training,
    TaskType,
)

from datasets import load_dataset, load_from_disk
# from utils import fix_model, fix_tokenizer
from itertools import chain
import numpy as np
import torch
import os


PROC = 8
TOKENIZER_BS = 100
MAX_SEQ_LEN = 2048


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=2, 
    lora_alpha=16, 
    lora_dropout=0.05,
    target_modules="model\.layers.*(q|k|v)_proj$",
)


tokenizer = ByT5Tokenizer()


MODEL_CONFIG = {
    "architectures": ["RoyalCanine"], 
    "bos_token_id": tokenizer.bos_token_id,
    "downsampling_rate": 4,
    "eos_token_id": tokenizer.eos_token_id,
    "hidden_act": "silu", 
    "hidden_size": 4096, 
    "intermediate_size": 11008, 
    "initializer_range": 0.02, 
    "max_sequence_length": 2048, 
    "model_type": "llama", 
    "num_attention_heads": 32, 
    "num_hidden_layers": 32, 
    "pad_token_id": tokenizer.pad_token_id,
    "rms_norm_eps": 1e-06, 
    "torch_dtype": "float16", 
    "transformers_version": "4.27.0.dev0",
    "vocab_size": tokenizer.vocab_size,
    "local_transformer_stride": 128,
    "upsampling_kernel_size": 4,
    "use_cache": False,
}


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(prompt)
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    return result


def chunk_examples(examples, max_length):
    input_ids_chunks = []
    attention_mask_chunks = []

    input_ids = list(chain(*examples["input_ids"]))
    attention_mask = list(chain(*examples["attention_mask"]))
    input_ids_chunks.extend(
        [
            input_ids[i : i + max_length]
            for i in range(0, len(input_ids), max_length)
        ]
    )
    attention_mask_chunks.extend(
        [
            attention_mask[i:i+max_length]
            for i in range(0, len(attention_mask), max_length)
        ]
    )
    return {
        "input_ids": input_ids_chunks,
        "attention_mask": attention_mask_chunks,
    }


def prepare_dataset(dataset):
    tokenized_dataset = dataset.map(lambda x: tokenize(x['text']),
                                    num_proc=PROC, remove_columns=["meta"])
    chunk_dataset = tokenized_dataset.map(lambda x: chunk_examples(x, MAX_SEQ_LEN), batched=True, batch_size=TOKENIZER_BS,
                                          num_proc=PROC, remove_columns=tokenized_dataset.column_names)
    return tokenized_dataset, chunk_dataset


if __name__ == "__main__":
    gradient_accumulation_steps = 8
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # tokenizer = fix_tokenizer(tokenizer)

    dataset = load_from_disk("filtered_tokenized_dataset")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # _, chunk_train_dataset = prepare_dataset(train_dataset)
    # _, chunk_val_dataset = prepare_dataset(val_dataset)

    # tokenizer.pad_token = tokenizer.unk_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False,
                                                    return_tensors="pt")

    model = LlamaWithCanineForCausalLM.from_pretrained('decapoda-research/llama-7b-hf', 
                                                       config=LlamaWithCanineConfig(**MODEL_CONFIG),
                                                       device_map=device_map)

    # model = fix_model(model, tokenizer, use_resize=False)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)

    for n, p in model.named_parameters():
        if not any(n.startswith(prefix) for prefix in ["base_model.model.model.layers", "base_model.model.model.norm.weight"]):
            p.requires_grad = True
        # print(n, p.requires_grad)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        warmup_steps=0,
        logging_steps=10,

        gradient_accumulation_steps=gradient_accumulation_steps,

        num_train_epochs=1,

        learning_rate=0.0003,
        # lr_scheduler_type="cosine",

        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,

        gradient_checkpointing=False,
        deepspeed="zero_config.json" if ddp else None,

        fp16=False,
        optim="adamw_torch",

        # torch_compile=False,

        logging_dir='./logs',
        output_dir='./13b_dataV2',
        adam_beta1=0.9,
        adam_beta2=0.95,
        save_total_limit=3,
        ddp_find_unused_parameters=False if ddp else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

