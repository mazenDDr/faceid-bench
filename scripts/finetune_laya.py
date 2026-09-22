"""Fine-tune Laya on "same person?" decisions from dev people, with Laya's own recipe.

Port of the training script in Laya's notebook (notebooks/laya_finetune_typed_decisions_2xT4
_kaggle.ipynb, laya 0.3.5) to one GPU: RLCD policy gradient with a proper scoring reward
(w_sph 0.75, w_rps 1.0) plus soft cross-entropy, 4 epochs, AdamW (encoder 2.5e-5, head 1e-4),
cosine schedule, noise sigma 0.4 -> 0.1, group size 4, fp16 + gradient checkpointing, then a
per-type temperature fit. Their effective batch of 64 (8 x 4 steps x 2 GPUs) becomes
8 x 8 steps on one GPU. Only dev people are used; test people stay untouched.

  python scripts/finetune_laya.py --out data/cache/laya_ft   (gpu-box, ~10-20 min)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from faceid_bench.data import identity_split, lfw_images
from faceid_bench.laya_decision import QUESTION, state
from faceid_bench.verify import PairScores

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/cache/laya_ft")
parser.add_argument("--pairs", type=int, default=4000, help="per class")
parser.add_argument("--variant", default="note")
parser.add_argument("--epochs", type=int, default=4)
args = parser.parse_args()

EPOCHS, MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = args.epochs, 8, 8, 4
LR_ENCODER, LR_HEAD, SIGMA_START, SIGMA_END = 2.5e-5, 1.0e-4, 0.4, 0.1
torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- data: balanced genuine / impostor pairs from dev people only -------------------------
cache = Path("data/cache/lfw")
emb = np.load(cache / "r50_w600k.npz")
det = np.load(cache / "detections_scrfd_10g_kps.npz")
index = {p: i for i, p in enumerate(emb["paths"])}
people = identity_split(lfw_images())["dev"]
names = sorted(people)
idx = np.array([index[p] for n in names for p in people[n]])
ident = np.array([k for k, n in enumerate(names) for _ in people[n]])
scores = PairScores(emb["embeddings"][idx], ident)

model_dir = snapshot_download("convaiinnovations/laya")
_fix_tokenizer_config(model_dir)
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096, max_len=1024, head_max_len=256)

(qdef,) = QUESTION.values()
q = {"t": "noul", "ins": qdef["instructions"], "crit": {}}
k = len(render_options(q))
items = []
for same, mask in ((True, scores.genuine_mask), (False, scores.impostor_mask)):
    r, c = np.nonzero(mask)
    pick = rng.choice(len(r), args.pairs, replace=False)
    for a, b in zip(r[pick], c[pick], strict=True):
        s = state(args.variant, scores.sim[a, b], det["rows"][idx[a]], det["rows"][idx[b]])
        seq, markers = build_sequence(tok, s, q, cfg["max_len"], cfg["head_max_len"])
        assert len(markers) == k
        target = [0.0, 1.0] if same else [1.0, 0.0]  # noul options: [false, true]
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES["noul"],
                "target": target,
                "label": int(same),
            }
        )
print(f"{len(items)} training decisions from {len(names)} dev people", flush=True)


def collate(chunk, pad_id):
    n, length = len(chunk), max(len(it["ids"]) for it in chunk)
    kmax = max(len(it["markers"]) for it in chunk)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(chunk):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        m = len(it["markers"])
        mpos[i, :m] = torch.tensor(it["markers"])
        mmask[i, :m] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    qtype = torch.tensor([it["qtype"] for it in chunk])
    return ids, att, mpos, mmask, target, qtype


# --- model and training loop (Laya's recipe) ------------------------------------------------
device = torch.device("cuda")
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.head_checkpointing = True
model.to(device).train()

enc = [p for n, p in model.named_parameters() if "encoder." in n]
head = [p for n, p in model.named_parameters() if "encoder." not in n]
optimizer = torch.optim.AdamW(
    [{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01
)
updates = (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, updates), eta_min=1e-6
)
scaler = torch.amp.GradScaler("cuda", enabled=True)
start, log = time.time(), []
for epoch in range(EPOCHS):
    random.Random(42 + epoch).shuffle(items)
    sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, EPOCHS - 1)
    optimizer.zero_grad(set_to_none=True)
    total, n_batches = 0.0, 0
    for b in range(0, len(items), MICRO_BATCH):
        ids, att, mpos, mmask, target, qtype = (
            t.to(device) for t in collate(items[b : b + MICRO_BATCH], tok.pad_token_id)
        )
        with torch.autocast("cuda", dtype=torch.float16):
            logits, act = model(ids, att, mpos, mmask, qtype)
        logits = logits.float()
        kk = mmask.sum(-1, keepdim=True).float()
        eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mmask
        eps = (eps - eps.sum(-1, keepdim=True) / kk) * mmask
        z = logits.detach().unsqueeze(0) + eps
        qd = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
        with torch.no_grad():
            reward = proper_reward(qd, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
            adv = reward - reward.mean(0, keepdim=True)
            adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma**2)
        loss_rl = -(adv * logp).mean()
        loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
        loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
        scaler.scale(loss).backward()
        n_batches += 1
        if n_batches % GRAD_ACCUM == 0 or b + MICRO_BATCH >= len(items):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * GRAD_ACCUM
    log.append(
        {"epoch": epoch + 1, "avg_loss": total / n_batches, "seconds": round(time.time() - start)}
    )
    print(log[-1], flush=True)

# --- temperature fit, as in the recipe (on a slice of the training decisions) ---------------
model.eval()
calib = items[::15][:400]
Z, T = [], []
with torch.no_grad():
    for b in range(0, len(calib), 16):
        ids, att, mpos, mmask, target, qtype = (
            t.to(device) for t in collate(calib[b : b + 16], tok.pad_token_id)
        )
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(ids, att, mpos, mmask, qtype)
        Z.append(logits.float()[:, :k].cpu())
        T.append(target[:, :k].cpu())
Z, T = torch.cat(Z), torch.cat(T)
log_t = torch.zeros(1, requires_grad=True)
opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)


def closure():
    opt.zero_grad()
    loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
    loss.backward()
    return loss


opt.step(closure)
noul_temperature = float(torch.clamp(log_t.exp(), 0.1, 10.0))
temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
temps[QTYPES["noul"]] = noul_temperature

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
save_file(
    {k2: v.half().contiguous().cpu() for k2, v in model.state_dict().items()},
    str(out / "model.safetensors"),
)
model.encoder.config.save_pretrained(str(out / "encoder"))
tok.save_pretrained(str(out / "tokenizer"))
cfg.update(fine_tuned=True, model_name="laya-faceid-note", temperature=temps)
cfg.pop("temperature_by_options", None)  # buckets were fit for the base model
(out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
summary = {
    "items": len(items),
    "epochs": log,
    "noul_temperature": noul_temperature,
    "variant": args.variant,
}
Path("outputs/laya").mkdir(parents=True, exist_ok=True)
Path("outputs/laya/finetune_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
