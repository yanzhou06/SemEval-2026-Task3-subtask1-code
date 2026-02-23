# -*- coding: utf-8 -*-
"""
how to run:
torchrun --nproc_per_node=8 qwen-train.py
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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


#tool func
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
        #给 0.0 占位
        df['Valence'] = 0.0
        df['Arousal'] = 0.0
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    else:
        raise ValueError("格式不对，必须有 'Quadruplet', 'Triplet', 'Aspect_VA' or 'Aspect'")
    return df

def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v, is_norm=False):
    pcc_v = pearson_safe(pred_v, gold_v)
    pcc_a = pearson_safe(pred_a, gold_a)
    gold_va = list(gold_v) + list(gold_a)
    pred_va = list(pred_v) + list(pred_a)

    def rmse_norm(gold_va, pred_va, is_normalization=True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        if is_normalization:
            return math.sqrt(sum(result) / len(gold_v)) / math.sqrt(128)
        return math.sqrt(sum(result) / len(gold_v))

    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    return {'PCC_V': pcc_v, 'PCC_A': pcc_a, 'RMSE_VA': rmse_va}

def pearson_safe(x, y):
    try:
        r = pearsonr(x, y)[0]
        if np.isnan(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0

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

def load_all_eng_train_dfs(target_train_path: str) -> pd.DataFrame:

    base_dir = os.path.dirname(target_train_path)
    files = [
        fname for fname in os.listdir(base_dir)
        if "train" in fname and fname.endswith(".jsonl")
    ]
    if not files:
        raise FileNotFoundError(f"No .jsonl files containing 'train' in name found in {base_dir}")

    dfs = []
    for fname in sorted(files):
        path = os.path.join(base_dir, fname)
        raw = load_jsonl(path)
        df = jsonl_to_df(raw)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


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
        inputs_embeds=None,  # for PGD
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
            loss_corr = neg_pearson_loss(y, labels)
            return loss_main + loss_cfg.get("pearson_lambda", 0.05) * loss_corr, y
        else:
            return loss_main, y


def neg_pearson_loss(pred, gold, eps=1e-8):
    losses = []
    for d in range(pred.size(-1)):
        x = pred[:, d]
        y = gold[:, d]
        vx = x - x.mean()
        vy = y - y.mean()
        num = (vx * vy).sum()
        den = vx.norm() * vy.norm() + eps
        r = num / den
        losses.append(1.0 - r)
    return torch.stack(losses).mean()

def try_enable_nonreentrant_gc(model):
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    except TypeError:
        model.gradient_checkpointing_enable()
        return False
    except Exception:
        return False

def is_gc_enabled(model):
    for attr in ["gradient_checkpointing", "is_gradient_checkpointing"]:
        if hasattr(model, attr):
            try:
                return bool(getattr(model, attr))
            except Exception:
                pass
    if hasattr(model, "config") and hasattr(model.config, "gradient_checkpointing"):
        try:
            return bool(model.config.gradient_checkpointing)
        except Exception:
            pass
    return False


#  DDP func
def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m

def maybe_no_sync(model, use_no_sync: bool):
    if isinstance(model, DDP) and use_no_sync:
        return model.no_sync()
    from contextlib import nullcontext
    return nullcontext()


# 对抗损失（PGD, L∞, 嵌入层）
def pgd_adversarial_loss(model, input_ids, attention_mask, labels, cfg):
    base = unwrap_model(model)
    gc_was_on = is_gc_enabled(base.backbone)
    if cfg.get("disable_ckpt_during_pgd", True) and gc_was_on:
        try:
            base.backbone.gradient_checkpointing_disable()
        except Exception:
            pass

    emb_layer = base.backbone.get_input_embeddings()
    with torch.no_grad():
        E = emb_layer(input_ids)
    E_fp32 = E.detach().to(torch.float32)

    steps = int(cfg["pgd_steps"])
    eps = float(cfg["pgd_eps"])
    alpha = float(cfg["pgd_alpha"]) if cfg["pgd_alpha"] is not None else (eps / max(1, steps))
    delta = torch.zeros_like(E_fp32, requires_grad=True)

    for _ in range(steps):
        adv_embeds = (E_fp32 + delta).to(E.dtype)
        loss_step, _ = base(
            inputs_embeds=adv_embeds,
            attention_mask=attention_mask,
            labels=labels,
            loss_cfg=dict(
                huber_beta=cfg["huber_beta"],
                add_neg_pearson=cfg["add_neg_pearson"],
                pearson_lambda=cfg["pearson_lambda"]
            )
        )
        grad = torch.autograd.grad(loss_step, delta, only_inputs=True, retain_graph=False, create_graph=False)[0]
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
        delta.requires_grad_(True)

    adv_embeds_final = (E_fp32 + delta).to(E.dtype).detach()
    loss_adv, _ = base(
        inputs_embeds=adv_embeds_final,
        attention_mask=attention_mask,
        labels=labels,
        loss_cfg=dict(
            huber_beta=cfg["huber_beta"],
            add_neg_pearson=cfg["add_neg_pearson"],
            pearson_lambda=cfg["pearson_lambda"]
        )
    )

    if cfg.get("disable_ckpt_during_pgd", True) and gc_was_on:
        try:
            try_enable_nonreentrant_gc(base.backbone)
        except Exception:
            pass

    return loss_adv


# train/val/infer
def train_epoch(model, dataloader, optimizer, scheduler, device, cfg):
    model.train()
    total = 0.0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    mse_consistency = nn.MSELoss()   # R-Drop 一致性
    accum = cfg["grad_accum"]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Train[rank{get_rank()}]")):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        use_no_sync = ((batch_idx % accum) != (accum - 1))
        with maybe_no_sync(model, use_no_sync):
            if cfg.get("enable_rdrop", False):
                loss1, y1 = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    loss_cfg=dict(
                        huber_beta=cfg["huber_beta"],
                        add_neg_pearson=cfg["add_neg_pearson"],
                        pearson_lambda=cfg["pearson_lambda"]
                    )
                )
                loss2, y2 = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    loss_cfg=dict(
                        huber_beta=cfg["huber_beta"],
                        add_neg_pearson=cfg["add_neg_pearson"],
                        pearson_lambda=cfg["pearson_lambda"]
                    )
                )
                loss_clean = 0.5 * (loss1 + loss2) + cfg["rdrop_alpha"] * mse_consistency(y1, y2)
            else:
                loss_clean, _ = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    loss_cfg=dict(
                        huber_beta=cfg["huber_beta"],
                        add_neg_pearson=cfg["add_neg_pearson"],
                        pearson_lambda=cfg["pearson_lambda"]
                    )
                )

            if cfg.get("enable_pgd", False) and cfg.get("pgd_eps", 0.0) > 0.0 and cfg.get("pgd_steps", 0) > 0:
                loss_adv = pgd_adversarial_loss(model, input_ids, attention_mask, labels, cfg)
                loss_total = loss_clean + cfg["pgd_lambda"] * loss_adv
            else:
                loss_total = loss_clean

            loss = loss_total / accum
            loss.backward()

        if cfg["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), cfg["grad_clip"])

        if (batch_idx % accum) == (accum - 1):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total += loss_total.item()
        step += 1

    return total / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, device, cfg):
    model.eval()
    total = 0.0
    for batch in tqdm(dataloader, desc=f"Eval[rank{get_rank()}]"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        loss, _ = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            loss_cfg=dict(
                huber_beta=cfg["huber_beta"],
                add_neg_pearson=cfg["add_neg_pearson"],
                pearson_lambda=cfg["pearson_lambda"]
            )
        )
        total += loss.item()
    return total / len(dataloader)


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


@torch.no_grad()
def eval_rmse_va_on_dev(model, dataloader, device):
    """
    注：函数名保留 dev 是为了复用已有代码，实际语义是 val。
    """
    pred_v, pred_a, gold_v, gold_a = predict_on_loader(model, dataloader, device, expect_labels=True)
    metrics = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
    rmse_va = metrics["RMSE_VA"]
    return rmse_va, metrics


# 线性校准
def fit_linear_calibrator(y_pred, y_true):
    lr = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    a = float(lr.coef_[0]); b = float(lr.intercept_)
    return a, b

def apply_calibrator(y, a, b):
    return np.clip(a * y + b, 1.0, 9.0)


# main
def main():
    local_rank = setup_distributed()
    world_size = get_world_size()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(CFG["save_dir"], exist_ok=True)
    set_seed(CFG["seed"])

    if is_main_process():
        print(f"[DDP] world_size={world_size}, per_gpu_bs={CFG['per_gpu_bs']}, grad_accum={CFG['grad_accum']}")


    data_dir = os.path.dirname(CFG["train_path"])
    if is_main_process():
        print("Loading training data from all {lang}_{domain}_train_*.jsonl and splitting 10% val from each ...")
        print(f"Data dir = {data_dir}")

    train_files = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        m = re.match(r"([^_]+)_([^_]+)_train_.*\.jsonl$", fname)
        if m:
            lang_i, domain_i = m.group(1), m.group(2)
            train_files.append((fname, lang_i, domain_i))

    if not train_files:
        raise FileNotFoundError(f"No train files matching pattern '{{lang}}_{{domain}}_train_*.jsonl' found in {data_dir}")

    all_train_splits = []   # train 文件的 90% 
    all_val_splits = []     # train 文件的 10% 
    target_train_splits = []  # 目标 lang/domain 的 90% 

    for fname, lang_i, domain_i in train_files:
        path = os.path.join(data_dir, fname)
        train_raw = load_jsonl(path)
        train_full_i = jsonl_to_df(train_raw)
        train_df_i, val_df_i = train_test_split(train_full_i, test_size=0.1, random_state=CFG["seed"])

        all_train_splits.append(train_df_i)
        all_val_splits.append(val_df_i)

        if (lang_i == CFG["lang"]) and (domain_i == CFG["domain"]):
            target_train_splits.append(train_df_i)

        if is_main_process():
            print(
                f"  - {fname} (lang={lang_i}, domain={domain_i}): "
                f"train_size={len(train_df_i)}, val_size={len(val_df_i)}"
            )

    all_train_full = pd.concat(all_train_splits, ignore_index=True)
    val_df = pd.concat(all_val_splits, ignore_index=True)

    if target_train_splits:
        domain_train_df = pd.concat(target_train_splits, ignore_index=True)
    else:
        domain_train_df = all_train_full.copy()
        if is_main_process():
            print(
                f"[WARN] No train file matched target lang/domain {CFG['lang']}/{CFG['domain']}; "
                f"use ALL train data as target-domain finetune set."
            )

    # 多领域预训练数据（Stage 1）
    if CFG.get("use_multi_domain_pretrain", True):
        all_train_for_pretrain = all_train_full
    else:
        all_train_for_pretrain = domain_train_df.copy()

    pretrain_df = all_train_for_pretrain.reset_index(drop=True)
    finetune_df = domain_train_df.reset_index(drop=True)

    if is_main_process():
        print(
            f"Pretrain train size (multi-domain, val held-out): {len(pretrain_df)}, "
            f"Finetune train size (target domain): {len(finetune_df)}, "
            f"Val size (from all train files): {len(val_df)}"
        )

    #Tokenizer
    if is_main_process():
        print("Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"], trust_remote_code=CFG["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #Backbone(Qwen)
    dtype = torch.bfloat16 if (CFG["use_bf16"] and torch.cuda.is_available()) else None
    backbone = AutoModel.from_pretrained(
        CFG["model_name"],
        dtype=dtype,
        trust_remote_code=CFG["trust_remote_code"]
    )
    backbone.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    nonreentrant_ok = try_enable_nonreentrant_gc(backbone)
    if is_main_process():
        print(f"Enable gradient checkpointing (non-reentrant ok? {nonreentrant_ok})")

    #LoRA
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
        if is_main_process():
            backbone.print_trainable_parameters()

    #Regressor
    base_model = LLMVARegressor(backbone, dropout=0.1, map_to_19=True)
    base_model.to(device)

    #DDP
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

    #Dataset & DataLoader
    pretrain_dataset = VADataset(pretrain_df, tokenizer, max_len=CFG["max_len"])
    finetune_dataset = VADataset(finetune_df, tokenizer, max_len=CFG["max_len"])
    val_dataset = VADataset(val_df, tokenizer, max_len=CFG["max_len"])

    pretrain_sampler = DistributedSampler(
        pretrain_dataset, num_replicas=world_size, rank=get_rank(),
        shuffle=True, drop_last=False, seed=CFG["seed"]
    )
    finetune_sampler = DistributedSampler(
        finetune_dataset, num_replicas=world_size, rank=get_rank(),
        shuffle=True, drop_last=False, seed=CFG["seed"]
    )

    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=CFG["per_gpu_bs"], sampler=pretrain_sampler,
        num_workers=0, pin_memory=True
    )
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=CFG["per_gpu_bs"], sampler=finetune_sampler,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG["eval_bs"], shuffle=False,
        num_workers=0, pin_memory=True
    )

    #分层 LR
    head_params = list(unwrap_model(model).reg_head.parameters())
    lora_params = [p for n, p in unwrap_model(model).named_parameters() if ("lora_" in n) and p.requires_grad]
    optim_groups = [
        {"params": head_params, "lr": CFG["lr_head"]},
        {"params": lora_params, "lr": CFG["lr_lora"]},
    ]
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=CFG["weight_decay"])

    grad_accum = CFG["grad_accum"]
    pretrain_steps_per_epoch = max(1, math.ceil(len(pretrain_loader) / grad_accum))
    finetune_steps_per_epoch = max(1, math.ceil(len(finetune_loader) / grad_accum))

    total_epochs = CFG["epochs"]
    pre_ratio = CFG.get("pretrain_ratio", 0.5)
    pre_epochs = max(1, int(round(total_epochs * pre_ratio)))
    ft_epochs = max(1, total_epochs - pre_epochs)
    # 防止 round 之后两边之和不等于 total
    if pre_epochs + ft_epochs != total_epochs:
        ft_epochs = total_epochs - pre_epochs

    if is_main_process():
        print(f"Total epochs = {total_epochs}, pretrain_epochs = {pre_epochs}, finetune_epochs = {ft_epochs}")
        print(f"Pretrain steps/epoch = {pretrain_steps_per_epoch}, Finetune steps/epoch = {finetune_steps_per_epoch}")

    total_training_steps = pre_epochs * pretrain_steps_per_epoch + ft_epochs * finetune_steps_per_epoch
    num_warmup = int(CFG["warmup_ratio"] * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=total_training_steps
    )

    #训练：Stage1 + Stage2
    best_rmse = float("inf")
    best_path = os.path.join(CFG["save_dir"], "best.pt")
    global_epoch_idx = 0

    # Stage 1
    for ep in range(1, pre_epochs + 1):
        global_epoch_idx += 1
        pretrain_sampler.set_epoch(CFG["seed"] + global_epoch_idx)
        tr = train_epoch(model, pretrain_loader, optimizer, scheduler, device, CFG)

        if is_main_process():
            val_loss = eval_epoch(model, val_loader, device, CFG)
            rmse_va, val_metrics = eval_rmse_va_on_dev(model, val_loader, device)
            print(
                f"[Pretrain Epoch {ep}/{pre_epochs} | Global {global_epoch_idx}/{total_epochs}] "
                f"train_loss={tr:.4f}  val_loss={val_loss:.4f}  "
                f"val_PCC_V={val_metrics['PCC_V']:.4f}  val_PCC_A={val_metrics['PCC_A']:.4f}  "
                f"val_RMSE_VA={rmse_va:.4f}"
            )
            if rmse_va < best_rmse:
                best_rmse = rmse_va
                torch.save({"model": unwrap_model(model).state_dict(), "cfg": CFG}, best_path)
                print(f"  -> saved best (val_RMSE_VA={best_rmse:.4f}) to {best_path}")
        barrier()

    # Stage 2
    for ep in range(1, ft_epochs + 1):
        global_epoch_idx += 1
        finetune_sampler.set_epoch(CFG["seed"] + global_epoch_idx)
        tr = train_epoch(model, finetune_loader, optimizer, scheduler, device, CFG)

        if is_main_process():
            val_loss = eval_epoch(model, val_loader, device, CFG)
            rmse_va, val_metrics = eval_rmse_va_on_dev(model, val_loader, device)
            print(
                f"[Finetune Epoch {ep}/{ft_epochs} | Global {global_epoch_idx}/{total_epochs}] "
                f"train_loss={tr:.4f}  val_loss={val_loss:.4f}  "
                f"val_PCC_V={val_metrics['PCC_V']:.4f}  val_PCC_A={val_metrics['PCC_A']:.4f}  "
                f"val_RMSE_VA={rmse_va:.4f}"
            )
            if rmse_va < best_rmse:
                best_rmse = rmse_va
                torch.save({"model": unwrap_model(model).state_dict(), "cfg": CFG}, best_path)
                print(f"  -> saved best (val_RMSE_VA={best_rmse:.4f}) to {best_path}")
        barrier()

    # 在 val 上做评估/校准（rank0） ----
    if is_main_process():
        ckpt = torch.load(best_path, map_location="cpu")
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        unwrap_model(model).to(device)

        # 在合并后的 val 上评估/校准
        pred_v, pred_a, gold_v, gold_a = predict_on_loader(model, val_loader, device, expect_labels=True)
        val_eval = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
        print("Val Eval (pre-calibration):", val_eval)

        cali = None
        if CFG["do_calibration"]:
            a_v, b_v = fit_linear_calibrator(np.array(pred_v), np.array(gold_v))
            a_a, b_a = fit_linear_calibrator(np.array(pred_a), np.array(gold_a))
            print(f"Calibration params -> V: a={a_v:.4f}, b={b_v:.4f} | A: a={a_a:.4f}, b={b_a:.4f}")

            pred_v_c = apply_calibrator(np.array(pred_v), a_v, b_v)
            pred_a_c = apply_calibrator(np.array(pred_a), a_a, b_a)
            val_eval_c = evaluate_predictions_task1(pred_a_c, pred_v_c, gold_a, gold_v)
            print("Val Eval (calibrated):", val_eval_c)

            cali = {"a_v": a_v, "b_v": b_v, "a_a": a_a, "b_a": b_a}
            with open(os.path.join(CFG["save_dir"], "calibration.json"), "w", encoding="utf-8") as f:
                json.dump(cali, f, ensure_ascii=False, indent=2)

        print("[TRAIN DONE] best.pt & (optional) calibration.json are ready. Use predict.py for dev inference/export.")

    barrier()
    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
