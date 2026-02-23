# -*- coding: utf-8 -*-
"""
run:
torchrun --nproc_per_node=8 qwen-infer.py
"""

import os
import re
import json
import math
import random
from typing import List, Dict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType


#CFG
CFG = dict(
    # 注：train_path用于确定 data_dir,实际训练会使用该目录下所有 {lang}_{domain}_train_*.jsonl
    train_path="./eng_restaurant_train_alltasks.jsonl",
    # 注：predict_path同目录的所有的 {lang}_{domain}_dev_*.jsonl都是测试集
    predict_path="./eng_restaurant_dev_task1.jsonl",

    subtask="subtask_1",
    task="task1",
    lang="eng",
    domain="restaurant",

    #multi domain pretrain
    # 在 train_path 所在目录下扫描 {lang}_{domain}_train_*.jsonl 做 Stage1 预训练
    use_multi_domain_pretrain=True,
    # 总 epoch 数 & 预训练占比
    epochs=10,
    pretrain_ratio=1.0,   # example: 0.5 表示一半 epoch 预训练，一半精调

    # 模型与训练
    model_name="/gemini/user/private/Qwen2.5-7B",
    # model_name="/gemini-1/space/ckpts/wangsq/others/model/qwen2.5/qwen2.5-14B-Instruct",
    trust_remote_code=True,
    use_lora=True,
    use_qlora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "W_pack"],

    max_len=128,
    #batch cfg
    per_gpu_bs=4,
    eval_bs=64,
    grad_accum=1,

    #lr&optimize
    lr_head=1e-3,
    lr_lora=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    huber_beta=0.5,
    add_neg_pearson=False,
    pearson_lambda=0.05,

    #精度
    use_bf16=True,
    grad_clip=1.0,
    seed=42,

    #R-Drop / PGD 
    enable_rdrop=True,
    rdrop_alpha=0.5,
    enable_pgd=True,
    pgd_eps=2e-2,
    pgd_steps=3,
    pgd_alpha=None,     # None => eps/steps
    pgd_lambda=0.5,

    disable_ckpt_during_pgd=True,  # PGD 内关闭 gradient checkpointing

    #infer&calibration
    do_calibration=True,
    save_dir="./qwen-preds",

    pred_filename="pred_{lang}_{domain}.jsonl"
)


# DDP func
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if ddp_is_initialized():
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not ddp_is_initialized():
        to = timedelta(minutes=30)
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=to,
                device_id=local_rank,
            )
        except TypeError:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=to,
            )
    return local_rank


# tools
def set_seed(seed: int = 42):
    rs = seed + get_rank()
    random.seed(rs)
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)

def load_jsonl(filepath: str) -> List[Dict]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def jsonl_to_df(data: List[Dict]) -> pd.DataFrame:
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df['Valence'] = 0.0
        df['Arousal'] = 0.0
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    else:
        raise ValueError("Invalid format: must include 'Quadruplet', 'Triplet', 'Aspect_VA' or 'Aspect'")
    return df

def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

def df_to_jsonl(df: pd.DataFrame, out_path: str):
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)
    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {"ID": gid, "Aspect_VA": []}
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def apply_calibrator(y, a, b):
    return np.clip(a * y + b, 1.0, 9.0)


# Dataset
class VADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"[ASPECT] {self.aspects[idx]} [/ASPECT]\n[TEXT] {self.texts[idx]} [/TEXT]"
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }
        return item


# 模型：Qwen + LoRA + mean-pooling + 区间映射
class LLMVARegressor(nn.Module):
    def __init__(self, backbone, dropout=0.1, map_to_19=True):
        super().__init__()
        self.backbone = backbone
        hs = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hs, 2)
        self.map_to_19 = map_to_19

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_cfg=None
    ):
        if inputs_embeds is not None:
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        H = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        h = (H.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = self.dropout(h)
        y = self.reg_head(h)
        if self.map_to_19:
            y = 1.0 + 8.0 * torch.sigmoid(y)

        if labels is None:
            return y

        loss_main = nn.SmoothL1Loss(beta=loss_cfg.get("huber_beta", 0.5))(y, labels)
        if loss_cfg.get("add_neg_pearson", False):
            return loss_main, y
        else:
            return loss_main, y


#gradient_checkpoint tool
def try_enable_nonreentrant_gc(model):
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    except TypeError:
        model.gradient_checkpointing_enable()
        return False
    except Exception:
        return False

def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m


@torch.no_grad()
def predict_on_loader(model, dataloader, device, expect_labels: bool = True):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(dataloader, desc=f"Predict[rank{get_rank()}]"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.append(y.cpu().numpy())
        if expect_labels and ("labels" in batch):
            labels.append(batch["labels"].numpy())
    preds = np.vstack(preds)
    if expect_labels and len(labels) > 0:
        labels = np.vstack(labels)
        return preds[:, 0], preds[:, 1], labels[:, 0], labels[:, 1]
    else:
        return preds[:, 0], preds[:, 1]


# main
def main():
    local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(CFG["save_dir"], exist_ok=True)
    set_seed(CFG["seed"])

    best_path = os.path.join(CFG["save_dir"], "best.pt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"best checkpoint not found: {best_path} (run train.py first)")

    if is_main_process():
        print(f"[LOAD] best checkpoint: {best_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"], trust_remote_code=CFG["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Backbone
    dtype = torch.bfloat16 if (CFG["use_bf16"] and torch.cuda.is_available()) else None
    backbone = AutoModel.from_pretrained(
        CFG["model_name"],
        dtype=dtype,
        trust_remote_code=CFG["trust_remote_code"]
    )
    backbone.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    try_enable_nonreentrant_gc(backbone)

    # LoRA
    if CFG["use_lora"]:
        lora_cfg = LoraConfig(
            r=CFG["lora_r"],
            lora_alpha=CFG["lora_alpha"],
            lora_dropout=CFG["lora_dropout"],
            target_modules=CFG["target_modules"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        backbone = get_peft_model(backbone, lora_cfg)

    # Regressor 
    base_model = LLMVARegressor(backbone, dropout=0.1, map_to_19=True)
    base_model.to(device)

    # DDP
    if ddp_is_initialized() and torch.cuda.is_available():
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        for k, v in (("gradient_as_bucket_view", True), ("static_graph", True)):
            try:
                ddp_kwargs[k] = v
            except TypeError:
                pass
        model = DDP(base_model, **ddp_kwargs)
    else:
        model = base_model

    # load state dict
    ckpt = torch.load(best_path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    unwrap_model(model).to(device)

    # load calibration
    cali = None
    cali_path = os.path.join(CFG["save_dir"], "calibration.json")
    if CFG["do_calibration"] and os.path.isfile(cali_path):
        with open(cali_path, "r", encoding="utf-8") as f:
            cali = json.load(f)
        if is_main_process():
            print(f"[LOAD] calibration: {cali_path} -> {cali}")

    # 对所有 dev 测试集进行预测（rank0）
    if is_main_process():
        data_dir = os.path.dirname(CFG["train_path"])
        dev_files = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            m = re.match(r"([^_]+)_([^_]+)_dev_.*\.jsonl$", fname)
            if m:
                lang_i, domain_i = m.group(1), m.group(2)
                dev_files.append((fname, lang_i, domain_i))

        print(f"Found {len(dev_files)} dev(test) files for prediction.")
        seen_out_paths = set()

        for fname, lang_i, domain_i in dev_files:
            dev_path = os.path.join(data_dir, fname)
            test_raw = load_jsonl(dev_path)
            test_df = jsonl_to_df(test_raw)

            test_dataset = VADataset(test_df, tokenizer, max_len=CFG["max_len"])
            test_loader = DataLoader(
                test_dataset, batch_size=CFG["eval_bs"], shuffle=False,
                num_workers=0, pin_memory=True
            )

            pred_v_pred, pred_a_pred = predict_on_loader(model, test_loader, device, expect_labels=False)

            if CFG["do_calibration"] and cali is not None:
                pred_v_pred = apply_calibrator(np.array(pred_v_pred), cali["a_v"], cali["b_v"])
                pred_a_pred = apply_calibrator(np.array(pred_a_pred), cali["a_a"], cali["b_a"])

            test_df["Valence"] = pred_v_pred
            test_df["Arousal"] = pred_a_pred

            out_name = CFG["pred_filename"].format(lang=lang_i, domain=domain_i)
            out_path = os.path.join(CFG["save_dir"], out_name)
            if out_path in seen_out_paths:
                print(f"[WARN] Multiple dev files for {lang_i}_{domain_i}, overwriting {out_name}")
            seen_out_paths.add(out_path)

            df_to_jsonl(test_df, out_path)
            print(f"Saved submission for {fname} to: {out_path}")

    barrier()
    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
