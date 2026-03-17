import os
import torch
import torch.nn as nn
from datetime import datetime

from data  import load_data_wiki          # our data pipeline (returns DataLoader + Vocab)
from model import BERTModel               # our from-scratch BERT

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

# ── Hyper-parameters ─────────────────────────────────────────────────────────
CSV_PATH        = 'dataset/wikitext103_paragraphs.csv'
VAL_CSV_PATH    = 'dataset/wikitext103_val_paragraphs.csv'
BATCH_SIZE      = 256     # reduced from 512 — MAX_LEN doubled so memory doubles
MAX_LEN         = 128     # captures ~75-80% of WikiText-103 pairs (was 64 → dropped ~60%)
NUM_HIDDENS     = 256     # wider hidden dim for richer representations
FFN_NUM_HIDDENS = 512     # 2× num_hiddens (standard BERT ratio)
NUM_HEADS       = 4       # 256 / 4 = 64 d_k per head (standard; d_k=16 at 128/8 was too small)
NUM_BLKS        = 4       # deeper stack to leverage 100× more data vs WikiText-2
DROPOUT         = 0.1     # dropout probability applied throughout the model
LR              = 1e-3    # peak learning rate
WARMUP_STEPS    = 1000    # linearly ramp LR 0 → LR over first 1000 steps, then cosine decay
NUM_EPOCHS      = 20      # 20 × 269K paragraphs ≫ 50 × WikiText-2 in total updates

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
log_path = f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

def log(msg):
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + '\n')

# ── Data ─────────────────────────────────────────────────────────────────────
print('loading data …')
train_iter, vocab = load_data_wiki(CSV_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN)

has_val = os.path.exists(VAL_CSV_PATH)
if has_val:
    val_iter, _ = load_data_wiki(VAL_CSV_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN)

print(f'vocab size : {len(vocab):,}   |   batches per epoch : {len(train_iter):,}')

# ── Model ─────────────────────────────────────────────────────────────────────
model = BERTModel(
    vocab_size      = len(vocab),
    num_hiddens     = NUM_HIDDENS,
    ffn_num_hiddens = FFN_NUM_HIDDENS,
    num_heads       = NUM_HEADS,
    num_blks        = NUM_BLKS,
    dropout         = DROPOUT,
    max_len         = MAX_LEN,
)
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'parameters : {num_params:,}')

# ── Loss functions ────────────────────────────────────────────────────────────
mlm_loss_fn = nn.CrossEntropyLoss(reduction='none')
nsp_loss_fn = nn.CrossEntropyLoss()


def run_batch(batch):
    """Forward pass. Returns (total_loss, mlm_loss, nsp_loss, mlm_acc, nsp_acc)."""
    tokens, segments, valid_lens, mlm_pos, mlm_weights, mlm_labels, nsp_labels = [
        x.to(device) for x in batch
    ]

    _, mlm_Y_hat, nsp_Y_hat = model(tokens, segments, valid_lens.float(), mlm_pos)

    # ── MLM ──────────────────────────────────────────────────────────────────
    V        = mlm_Y_hat.shape[-1]
    mlm_loss = mlm_loss_fn(mlm_Y_hat.reshape(-1, V), mlm_labels.reshape(-1))
    mlm_loss = (mlm_loss * mlm_weights.reshape(-1)).sum() / mlm_weights.sum()

    # MLM accuracy: correct predictions at real (non-padded) mask positions
    mlm_preds   = mlm_Y_hat.argmax(dim=-1)                    # (B, num_preds)
    mask        = mlm_weights.bool()                           # True at real predictions
    mlm_correct = (mlm_preds[mask] == mlm_labels[mask]).sum().item()
    mlm_total   = mask.sum().item()

    # ── NSP ──────────────────────────────────────────────────────────────────
    nsp_loss    = nsp_loss_fn(nsp_Y_hat, nsp_labels)
    nsp_correct = (nsp_Y_hat.argmax(dim=-1) == nsp_labels).sum().item()
    nsp_total   = nsp_labels.shape[0]

    total_loss = mlm_loss + nsp_loss
    return total_loss, mlm_loss.item(), nsp_loss.item(), mlm_correct, mlm_total, nsp_correct, nsp_total


def evaluate(data_iter):
    """Run one pass over data_iter with no gradient updates. Returns metrics dict."""
    model.eval()
    totals = dict(loss=0.0, mlm_loss=0.0, nsp_loss=0.0,
                  mlm_correct=0, mlm_total=0, nsp_correct=0, nsp_total=0, batches=0)
    with torch.no_grad():
        for batch in data_iter:
            loss, ml, nl, mc, mt, nc, nt = run_batch(batch)
            totals['loss']        += loss.item()
            totals['mlm_loss']    += ml
            totals['nsp_loss']    += nl
            totals['mlm_correct'] += mc
            totals['mlm_total']   += mt
            totals['nsp_correct'] += nc
            totals['nsp_total']   += nt
            totals['batches']     += 1
    n = totals['batches']
    return {
        'loss'    : totals['loss']        / n,
        'mlm_loss': totals['mlm_loss']    / n,
        'nsp_loss': totals['nsp_loss']    / n,
        'mlm_acc' : totals['mlm_correct'] / totals['mlm_total'],
        'nsp_acc' : totals['nsp_correct'] / totals['nsp_total'],
    }


# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── LR scheduler: linear warmup then cosine decay ────────────────────────────
total_steps = NUM_EPOCHS * len(train_iter)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Log header ────────────────────────────────────────────────────────────────
header = (f"epoch | lr       | "
          f"trn_loss | trn_mlm_loss | trn_nsp_loss | trn_mlm_acc | trn_nsp_acc"
          + (" | val_loss | val_mlm_loss | val_nsp_loss | val_mlm_acc | val_nsp_acc" if has_val else ""))
log(f"Run started : {datetime.now()}")
log(f"Params      : {num_params:,}")
log(f"Train CSV   : {CSV_PATH}")
log(f"Val CSV     : {VAL_CSV_PATH if has_val else 'N/A'}")
log("-" * len(header))
log(header)
log("-" * len(header))

# ── Training loop ─────────────────────────────────────────────────────────────
print('\nstarting training …\n')
global_step = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    totals = dict(loss=0.0, mlm_loss=0.0, nsp_loss=0.0,
                  mlm_correct=0, mlm_total=0, nsp_correct=0, nsp_total=0, batches=0)

    for batch in train_iter:
        optimizer.zero_grad()
        loss, ml, nl, mc, mt, nc, nt = run_batch(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        totals['loss']        += loss.item()
        totals['mlm_loss']    += ml
        totals['nsp_loss']    += nl
        totals['mlm_correct'] += mc
        totals['mlm_total']   += mt
        totals['nsp_correct'] += nc
        totals['nsp_total']   += nt
        totals['batches']     += 1
        global_step           += 1

    n          = totals['batches']
    trn_loss   = totals['loss']        / n
    trn_ml     = totals['mlm_loss']    / n
    trn_nl     = totals['nsp_loss']    / n
    trn_mlm_acc = totals['mlm_correct'] / totals['mlm_total']
    trn_nsp_acc = totals['nsp_correct'] / totals['nsp_total']
    current_lr  = scheduler.get_last_lr()[0]

    row = (f"{epoch:5d} | {current_lr:.2e} | "
           f"{trn_loss:8.4f} | {trn_ml:12.4f} | {trn_nl:12.4f} | "
           f"{trn_mlm_acc:11.4f} | {trn_nsp_acc:11.4f}")

    if has_val:
        v = evaluate(val_iter)
        row += (f" | {v['loss']:8.4f} | {v['mlm_loss']:12.4f} | {v['nsp_loss']:12.4f} | "
                f"{v['mlm_acc']:11.4f} | {v['nsp_acc']:11.4f}")
        model.train()

    log(row)

    if epoch % 5 == 0:
        ckpt_path = f'/home/jovyan/nanoBERT/checkpoints/bert_epoch{epoch}.pt'
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'vocab_size'  : len(vocab),
        }, ckpt_path)
        log(f'  → checkpoint saved to {ckpt_path}')

log("-" * len(header))
log(f"Run finished : {datetime.now()}")
print('\ntraining complete.')
