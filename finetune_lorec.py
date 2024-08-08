import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import time
from transformers import set_seed
from dataset import generate_and_tokenize_prompt
from modeling_lorec_trash_v1 import Lorec
from modeling_lorec_trash_v2 import Lorec_moe
from dataset import get_text_and_ctr_dataset, LorecDataCollator
from ctr_base.config import Config
from functools import partial
from utils import lora_tsne, get_ctr_config

import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="/data/zhujiachen/llm/models/lorec")
# parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str, default="/data/zhujiachen/llm/models/vicuna-7b-v1.5")
parser.add_argument("--model_name", type=str, default="vicuna-13b")
parser.add_argument("--ctr_model_path", type=str, default="/data/zhujiachen/llm/models/ctr_model/BookCrossing/din/model.pt")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=256)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--test_range", type=str, default='all')
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--wd", type=float, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--only_eval", action='store_true')
parser.add_argument("--dataset", type=str, default="BookCrossing")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--debugging", type=bool, default=False)

#model args
parser.add_argument("--mode", type=str, default='origin')
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--layers", type=str, default="[q,v]")
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--enable_softmax", action="store_true")
parser.add_argument("--enable_sqrt_after_softmax", action="store_true")
parser.add_argument("--head", type=str, default='lm')
parser.add_argument("--init_method", type=str, default='kaiming')
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ret", action="store_true")
parser.add_argument("--load_rella", action="store_true")
parser.add_argument("--all_init_one", action="store_true")
parser.add_argument("--lora_assemble_num", type=int, default=8)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--routing_mode", type=str, default='soft')
parser.add_argument("--which_in_gate", type=str, default='None')
parser.add_argument("--ctr_dropout", type=float, default=0.1)
parser.add_argument("--agg_pt", type=str, default='w')
parser.add_argument("--ctr_out_layer", type=str, default='final')
parser.add_argument("--ctr_K", type=int, default=30)

# Here are args of prompt
parser.add_argument("--K", type=int, default=30)
parser.add_argument("--train_type", type=str, default="simple")
parser.add_argument("--test_type", type=str, default="simple")

args = parser.parse_args()

assert args.train_type in ["simple", "sequential", "mixed", "high","all"]
assert args.test_type in ["simple", "sequential", "high"]
assert args.dataset in ["ml-1m", "BookCrossing", "GoodReads", "AZ-Toys", "ml-25m"]

data_path = f"/data/zhujiachen/Datasets/{args.dataset}/benchmark_proc_data/data"

t1 = time.time()
if args.layers == "[up,down,gate]":
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = 2
else:
    if args.K <= 15:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
    elif args.K <= 40:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
    else:
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1


print('*'*70)
print(args)
print('*'*70)

transformers.set_seed(args.seed)

if args.train_type == "mixed":
    print(f"Shot: {args.train_size}")
    args.train_size *= 2
    print(f"Samples used: {args.train_size}")

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
# optimized for RTX 4090. for larger GPUs, increase some of these?

BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
EPOCHS = args.epochs  # we don't always need 3 tbh
LEARNING_RATE = args.lr  # the Karpathy constant
CUTOFF_LEN = 2048  # 256 accounts for about 96% of the data
LORA_R = args.lora_r
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = args.lora_dropout
VAL_SET_SIZE = args.val_size #2000
USE_8bit = True

if USE_8bit is True:
    warnings.warn("If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2")
        
DATA_PATH = {
    "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}_sampled.json"]), 
    # "train": '/'.join([data_path, f"train/train_{args.K}_{args.train_type}.json"]), 
    # "val": '/'.join([args.data_path, f"valid/valid_{args.K}_{args.test_type}_sampled.json"]),
    "test": '/'.join([data_path, f"test/test_{args.K}_{args.test_type}.json"])
}
if args.train_type == "all" or args.dataset == 'ml-25m':
    DATA_PATH["train"] = '/'.join([data_path, f"train/train_{args.K}_{args.train_type}.json"])

OUTPUT_DIR = args.output_path #"lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

print(world_size)
print(ddp)

# llama_config = {"model_name": "vicuna-7b", "model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 4096, 'layer_num': 32}
# llama_config = {"model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 5120, 'layer_num': 40}


if args.model_name == "vicuna-7b":
    args.model_path = "/data/zhujiachen/llm/models/vicuna-7b-v1.5"
    llama_config = {"model_name": "vicuna-7b", "model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 4096, 'intermediate_dim':11008, 'layer_num': 32}
    # args.per_device_train_batch_size *=2
    # args.per_device_eval_batch_size *=2
elif args.model_name == "vicuna-13b":
    args.model_path = "/data/zhujiachen/llm/models/vicuna"
    llama_config = {"model_name": "vicuna-13b", "model_path": args.model_path, "load_in_8bit": USE_8bit, "device_map":device_map, 'hidden_dim': 5120, 'intermediate_dim':13824, 'layer_num': 40}

print(args.per_device_train_batch_size)
print(args.per_device_eval_batch_size)
print(llama_config)

ctr_config = get_ctr_config(args)
ctr_config = Config.from_dict(ctr_config)


TARGET_MODULES = args.layers[1:-1].split(',')

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
lora_config.enable_softmax = args.enable_softmax
lora_config.enable_sqrt_after_softmax = args.enable_sqrt_after_softmax
assert not (not lora_config.enable_softmax and lora_config.enable_sqrt_after_softmax), f"`enable_softmax` must be True, if `enable_sqrt_after_softmax` is True."

lora_config.all_init_one = args.all_init_one
lora_config.rella_model_pt = f'/data/zhujiachen/llm/models/rella_model/{args.dataset}/{args.model_name}/pytorch_model.bin' if args.load_rella else None
# assert not (not lora_config.enable_softmax and lora_config.enable_sqrt_after_softmax), f"`rella_model_pt` must be True, if `all_init_one` is True."

lora_config.routing_mode = args.routing_mode
lora_config.topk = 2
# assert not (not lora_config.enable_softmax and lora_config.enable_sqrt_after_softmax), f"`rella_model_pt` must be True, if `all_init_one` is True."

if args.K >20 or args.lora_assemble_num > 8 :
    use_gradient_checkpointing = True
else:
    use_gradient_checkpointing = False

if args.agg_pt == 'w': #//weight
    model = Lorec(llama_config, lora_config, ctr_config, mode=args.mode, head=args.head, init_method=args.init_method,temperature=args.temperature,which_in_gate=args.which_in_gate,USE_GC = use_gradient_checkpointing)
elif args.agg_pt == 'o': #output
    print(args.agg_pt)
    model = Lorec_moe(llama_config, lora_config, ctr_config, mode=args.mode, head=args.head, init_method=args.init_method,temperature=args.temperature,which_in_gate=args.which_in_gate)


if args.resume:
    model_dict = torch.load('/data/zhujiachen/llm/ml-25m_lorec/_Dataset_ml-25m_CUDA_NUM_2_test_range_0:30000_True_False_layers_[q,v]_lr_0.0001_shot_90000_sequential_sequential_K_30_10_bs128_wd_0.0_model_vicuna-7b_mode_ctr_loradropout_0.0_lora_r_8_init_kaiming__load_rella__False_all_init_one_False_lora_assemble_num_16/checkpoint-1406/pytorch_model.bin')
    model.load_state_dict(model_dict,strict=False)

tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, 
    add_eos_token=True, 
)

# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)


tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


CTR_DATA_PATH = f"/data/zhujiachen/Datasets/{args.dataset}/benchmark_proc_data"
if args.dataset == 'ml-25m':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "ml-25m", "sample_num": 90000, "hist_len": args.ctr_K, 'ret':args.ret}
elif args.dataset == 'BookCrossing':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "BookCrossing", "sample_num": 10000, "hist_len": args.ctr_K, 'ret':args.ret}
elif args.dataset == 'ml-1m':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "ml-1m", "sample_num": 70000, "hist_len" : args.ctr_K, 'ret':args.ret}
elif args.dataset == 'GoodReads':
    data_config = {"item_field_idx": ctr_config.item_field_idx, "dataset_name": "GoodReads", "sample_num": 70000, "hist_len" : args.ctr_K, 'ret':args.ret}


data = get_text_and_ctr_dataset(
    TEXT_DATA_PATH=DATA_PATH, 
    CTR_DATA_PATH=CTR_DATA_PATH, 
    train_size=args.train_size, 
    train_type=args.train_type,
    test_range=args.test_range,
    tokenizer=tokenizer, 
    CUTOFF_LEN=CUTOFF_LEN, 
    data_config=data_config,
)
train_data = data['train']
test_data = data['test']
# data = load_dataset("json", data_files=DATA_PATH)
# data["train"] = data["train"].select(range(args.train_size))

# test_data = data['test'].map(partial(generate_and_tokenize_prompt,tokenizer=tokenizer, CUTOFF_LEN=CUTOFF_LEN))
# # val_data = data["val"].map(generate_and_tokenize_prompt)
# train_data = data["train"].map(partial(generate_and_tokenize_prompt,tokenizer=tokenizer, CUTOFF_LEN=CUTOFF_LEN))
print("Data processed.")
# data["test"] = data["test"].select(range(args.train_size))
# print(args.debugging,'aaaaaaa')
# if args.debugging:
#     data["test"] = data["test"].select(range(args.train_size))

# data["val"] = data["val"].select(range(50))
print(data['train'])
print(data['test'])
print("Data loaded.")



now_max_steps = max((len(data["train"])) // BATCH_SIZE * EPOCHS, EPOCHS)
if args.resume_from_checkpoint:
    if args.lora_remote_checkpoint is not None:
        snapshot_download(repo_id=args.lora_remote_checkpoint, allow_patterns=["*.pt", "*.bin", "*.json"], local_dir=args.resume_from_checkpoint)
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None  # So the trainer won't try loading its state
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    
    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    
    if os.path.exists(train_args_path):
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps

print(MAX_STEPS)
# print("Load lora weights")
# adapters_weights = torch.load(os.path.join("lora-Vicuna/checkpoint-2", "pytorch_model.bin"))
# set_peft_model_state_dict(model, adapters_weights)
# print("lora load results")

# model.print_trainable_parameters()


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    if args.mode == "origin" or args.head == 'lm':
        labels_index = torch.argwhere(torch.bitwise_or(labels == 3869, labels == 1939))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 1939, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
        prob = torch.softmax(logits, dim=-1)
        return prob[:, 1], gold
    else:
        if args.head == 'ctr2':
            logits = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
            gold = torch.where(labels[:, -2] == 1939, 0, 1)
        elif args.head == 'ctr1':
            gold = torch.where(labels[:, -2] == 1939, 0, 1)
    return logits, gold

print(args.dataset)
# ctr_model load
if args.mode in ['ctr', 'vera']:
    if args.ret:
        args.ctr_model_path = f"/data/zhujiachen/llm/models/ctr_model/{args.dataset}/sim/hist_len_{args.ctr_K}/model.pt"
    else:
        args.ctr_model_path = f"/data/zhujiachen/llm/models/ctr_model/{args.dataset}/din/hist_len_{args.ctr_K}/model.pt"
    state_dict = torch.load(args.ctr_model_path)
    model.CTR_model.load_state_dict(state_dict)

print(args.dataset)
print(args.ctr_model_path)


# lora_tsne(model,llama_config)
set_seed(42)
model.LM.config.use_cache = False

class StopTrainingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # stop on the first epoch
        control.should_training_stop = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        # max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=False,
        logging_strategy="steps", 
        logging_steps=1,
        # evaluation_strategy="epoch" if VAL_SET_SIZE > 0 else "no",
        evaluation_strategy="epoch",
        # evaluation_strategy="no",
        save_strategy="epoch",
        save_safetensors = False,
        # save_strategy="epoch",
        # eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=30,
        # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        load_best_model_at_end=False,
        metric_for_best_model="eval_auc",
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
        weight_decay=args.weight_decay,
    ),
    data_collator=LorecDataCollator(tokenizer=tokenizer),
    # data_collator=transformers.DataCollatorForSeq2Seq(
    #     tokenizer, return_tensors="pt", padding='longest'
    # ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    callbacks = [] if args.dataset == 'BookCrossing' else [StopTrainingCallback()],
)
# print(12)
# if args.use_lora:
#     old_state_dict = model.state_dict
#     model.state_dict = (
#         lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
#     ).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    print(1)
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

print("Start training...")

set_seed(42)
# print(trainer.evaluate(eval_dataset=test_data))
# exit(0)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# model.save_pretrained(OUTPUT_DIR)


print("time1")
print(time.time()- t1)
print((time.time()-t1) / 3600)