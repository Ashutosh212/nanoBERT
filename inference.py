"""
inference.py  —  two quick tests to see how well nanoBERT learned:

  1. MLM  — fill in the [MASK] token in a sentence you type
  2. NSP  — give two sentences, model says whether B follows A

Usage:
    python3 inference.py                          # loads bert_epoch50.pt by default
    python3 inference.py --ckpt bert_epoch30.pt   # pick a specific checkpoint
"""

import argparse
import torch
import torch.nn.functional as F

from data  import load_data_wiki, MASK, CLS, SEP, PAD   # reuse vocab builder
from model import BERTModel

# ── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',    default='checkpoints/bert_epoch50.pt',
                    help='checkpoint file saved by train.py')
parser.add_argument('--csv',     default='dataset/wikitext2_paragraphs.csv',
                    help='same CSV used during training (needed to rebuild the vocab)')
parser.add_argument('--max_len', type=int, default=64)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Rebuild vocab (must match what training saw) ──────────────────────────────
print('rebuilding vocab from CSV …')
_, vocab = load_data_wiki(args.csv, batch_size=1, max_len=args.max_len)
print(f'vocab size: {len(vocab):,}\n')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.ckpt, map_location=device)

model = BERTModel(
    vocab_size       = len(vocab),
    num_hiddens      = 128,
    ffn_num_hiddens  = 256,
    num_heads        = 2,
    num_blks         = 2,
    dropout          = 0.0,          # no dropout at inference time
    max_len          = args.max_len,
)
model.load_state_dict(ckpt['model_state'])   # restore trained weights
model.to(device)
model.eval()                                  # switch off dropout / train-time behaviour
print(f'loaded checkpoint: {args.ckpt}  (epoch {ckpt["epoch"]})\n')


# ══════════════════════════════════════════════════════════════════════════════
# Helper: tokenise a raw string into word tokens (lower-case, space-split)
# This must match how data.py builds sentences so vocab look-ups work correctly
# ══════════════════════════════════════════════════════════════════════════════

def tokenise(text: str):
    # lowercase everything but preserve special tokens like [MASK] exactly as-is
    return [t if t.startswith('[') and t.endswith(']') else t.lower()
            for t in text.strip().split()]


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Masked Language Modelling
#   Give a sentence with one [MASK] placeholder.
#   The model should predict the most likely original token.
# ══════════════════════════════════════════════════════════════════════════════

def predict_mlm(sentence: str, top_k: int = 5):
    """
    sentence : plain text with exactly one '[MASK]' placeholder
               example: "the dog sat on the [MASK]"
    top_k    : how many candidate tokens to show
    """
    words = tokenise(sentence)

    # wrap with [CLS] … [SEP] just like training
    tokens   = [CLS] + words + [SEP]                 # list of string tokens
    segments = [0] * len(tokens)                     # all sentence-A (single sentence)

    # find which position holds the [MASK] token
    try:
        mask_idx = tokens.index(MASK)
    except ValueError:
        print('  ✗ no [MASK] found in input — please include the word [MASK]')
        return

    # convert tokens to ids; unknown words map to [UNK]
    token_ids = [vocab[t] for t in tokens]

    # pad to max_len
    valid_len = len(token_ids)
    token_ids += [vocab[PAD]] * (args.max_len - valid_len)
    segments  += [0]          * (args.max_len - len(segments))

    # build tensors with a batch dimension of 1
    t_tokens   = torch.tensor([token_ids],  dtype=torch.long).to(device)
    t_segments = torch.tensor([segments],   dtype=torch.long).to(device)
    t_valid    = torch.tensor([valid_len],  dtype=torch.float).to(device)
    t_pos      = torch.tensor([[mask_idx]], dtype=torch.long).to(device)  # predict only the mask

    with torch.no_grad():
        _, mlm_out, _ = model(t_tokens, t_segments, t_valid, t_pos)
        # mlm_out shape: (1, 1, vocab_size) — one prediction at the mask position
        logits = mlm_out[0, 0]                          # (vocab_size,)
        probs  = F.softmax(logits, dim=-1)              # convert to probabilities

    # grab top-k predicted tokens
    top_probs, top_ids = probs.topk(top_k)

    print(f'  input : {sentence}')
    print(f'  top-{top_k} predictions for [MASK]:')
    for rank, (idx, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        token = vocab.idx_to_token[idx]
        print(f'    {rank}. "{token}"  —  {prob*100:.1f}%')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Next Sentence Prediction
#   Give two sentences; model says if B is the real continuation of A.
# ══════════════════════════════════════════════════════════════════════════════

def predict_nsp(sent_a: str, sent_b: str):
    """
    sent_a, sent_b : plain text sentences (no special tokens needed)
    """
    tokens_a = tokenise(sent_a)
    tokens_b = tokenise(sent_b)

    # build [CLS] A [SEP] B [SEP] exactly as in training
    tokens   = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
    segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    # truncate if somehow too long
    tokens   = tokens[:args.max_len]
    segments = segments[:args.max_len]

    valid_len = len(tokens)
    token_ids = [vocab[t] for t in tokens]
    token_ids += [vocab[PAD]] * (args.max_len - valid_len)
    segments  += [0]          * (args.max_len - len(segments))

    t_tokens   = torch.tensor([token_ids], dtype=torch.long).to(device)
    t_segments = torch.tensor([segments],  dtype=torch.long).to(device)
    t_valid    = torch.tensor([valid_len], dtype=torch.float).to(device)

    with torch.no_grad():
        # pass pred_positions=None → model skips MLM head, only returns NSP logits
        _, _, nsp_out = model(t_tokens, t_segments, t_valid, pred_positions=None)
        probs = F.softmax(nsp_out[0], dim=-1)   # (2,)  index 0 = not-next, 1 = is-next

    is_next_prob  = probs[1].item()
    not_next_prob = probs[0].item()
    verdict = 'IS next sentence' if is_next_prob > 0.5 else 'NOT next sentence'

    print(f'  sentence A : {sent_a}')
    print(f'  sentence B : {sent_b}')
    print(f'  verdict    : B {verdict}')
    print(f'  confidence : is-next {is_next_prob*100:.1f}%  |  not-next {not_next_prob*100:.1f}%')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Run some demo examples
# ══════════════════════════════════════════════════════════════════════════════

print('=' * 60)
print('TEST 1 — Masked Language Modelling')
print('=' * 60)

predict_mlm('the dog sat on the [MASK]')
predict_mlm('he was born in [MASK] and grew up in the city')
predict_mlm('the [MASK] played a crucial role in the war')
predict_mlm('scientists discovered a new [MASK] in the laboratory')

print('=' * 60)
print('TEST 2 — Next Sentence Prediction')
print('=' * 60)

# should predict IS next
predict_nsp(
    'the film was directed by steven spielberg',
    'it was released in the summer of 1993'
)
# should predict NOT next
predict_nsp(
    'the cat sat on the mat',
    'the stock market fell sharply on monday'
)
# another real pair
predict_nsp(
    'he studied mathematics at the university',
    'after graduating he became a professor'
)
# another random pair
predict_nsp(
    'the army advanced through the mountains',
    'she enjoyed baking cookies on sundays'
)
