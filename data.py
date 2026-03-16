import csv
import math
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# special tokens
PAD, CLS, SEP, MASK, UNK = '[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'
SPECIALS = (PAD, CLS, SEP, MASK, UNK)


# ── Step 1: read CSV → list of paragraphs (each paragraph = list of sentences) ─

def _read_wiki(csv_path):
    paragraphs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['paragraph'].lower()
            # split on ' . ' (period surrounded by spaces, as WikiText-2 is pre-tokenised)
            sentences = [s.strip().split() for s in text.split(' . ') if s.strip()]
            if len(sentences) >= 2:
                paragraphs.append(sentences)
    random.shuffle(paragraphs)
    return paragraphs


# ── Vocabulary ──────────────────────────────────────────────────────────────────

class Vocab:
    def __init__(self, paragraphs, min_freq=2):
        counter = Counter(
            token
            for para in paragraphs
            for sent in para
            for token in sent
        )
        tokens = list(SPECIALS) + [
            tok for tok, freq in counter.items() if freq >= min_freq
        ]
        self.token_to_idx = {tok: i for i, tok in enumerate(tokens)}
        self.idx_to_token = tokens

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[UNK])


# ── Step 3: MLM — 15 % masking with 80 / 10 / 10 rule ─────────────────────────

def _get_mlm_data(tokens, vocab):
    """
    Returns (masked_tokens, pred_positions, mlm_label_ids).
    pred_positions and mlm_label_ids have the same length (<= round(len*0.15)).
    """
    candidates = [i for i, t in enumerate(tokens) if t not in SPECIALS]
    random.shuffle(candidates)
    num_preds = max(1, round(len(tokens) * 0.15))
    pred_positions = sorted(candidates[:num_preds])

    masked = list(tokens)
    mlm_labels = []
    for pos in pred_positions:
        mlm_labels.append(vocab[tokens[pos]])   # original token id
        r = random.random()
        if r < 0.8:
            masked[pos] = MASK
        elif r < 0.9:
            # random token from real vocabulary (skip the 5 specials)
            masked[pos] = vocab.idx_to_token[random.randint(5, len(vocab) - 1)]
        # else: leave unchanged (10 %)

    return masked, pred_positions, mlm_labels


# ── Step 2: NSP — build (tokens, segments, is_next) triples ────────────────────

def _get_nsp_data(paragraphs, vocab, max_len):
    all_sentences = [sent for para in paragraphs for sent in para]
    examples = []

    for para in paragraphs:
        for i in range(len(para) - 1):
            tokens_a = para[i]

            if random.random() < 0.5:
                is_next, tokens_b = True, para[i + 1]
            else:
                is_next, tokens_b = False, random.choice(all_sentences)

            # +3 for [CLS] … [SEP] … [SEP]
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue

            tokens   = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
            segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            examples.append((tokens, segments, is_next))

    return examples


# ── Step 4: MLM + padding → fixed-length tensors ───────────────────────────────

def _pad_bert_inputs(examples, max_len, max_num_mlm_preds, vocab):
    pad_id = vocab[PAD]

    all_token_ids    = []
    all_segments     = []
    all_valid_lens   = []
    all_mlm_pos      = []
    all_mlm_weights  = []
    all_mlm_labels   = []
    all_nsp_labels   = []

    for tokens, segments, is_next in examples:
        masked, pred_pos, mlm_labels = _get_mlm_data(tokens, vocab)

        valid_len  = len(masked)
        token_ids  = [vocab[t] for t in masked]
        num_preds  = len(pred_pos)

        # pad sequences to max_len
        token_ids += [pad_id] * (max_len - valid_len)
        segments  += [0]      * (max_len - len(segments))

        # pad MLM slots to max_num_mlm_preds
        mlm_weights  = [1.0] * num_preds + [0.0] * (max_num_mlm_preds - num_preds)
        pred_pos     = pred_pos           + [0]   * (max_num_mlm_preds - num_preds)
        mlm_labels   = mlm_labels         + [pad_id] * (max_num_mlm_preds - num_preds)

        all_token_ids.append(token_ids)
        all_segments.append(segments)
        all_valid_lens.append(valid_len)
        all_mlm_pos.append(pred_pos)
        all_mlm_weights.append(mlm_weights)
        all_mlm_labels.append(mlm_labels)
        all_nsp_labels.append(int(is_next))

    return (
        torch.tensor(all_token_ids,   dtype=torch.long),
        torch.tensor(all_segments,    dtype=torch.long),
        torch.tensor(all_valid_lens,  dtype=torch.long),
        torch.tensor(all_mlm_pos,     dtype=torch.long),
        torch.tensor(all_mlm_weights, dtype=torch.float),
        torch.tensor(all_mlm_labels,  dtype=torch.long),
        torch.tensor(all_nsp_labels,  dtype=torch.long),
    )


# ── Dataset wrapper ─────────────────────────────────────────────────────────────

class WikiDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


# ── Public API ──────────────────────────────────────────────────────────────────

def load_data_wiki(csv_path, batch_size=512, max_len=64):
    """
    Returns (DataLoader, Vocab).

    Each batch is a 7-tuple:
        token_ids    [B, max_len]            long
        segments     [B, max_len]            long   (0 = sent-A, 1 = sent-B)
        valid_lens   [B]                     long
        mlm_pos      [B, max_num_mlm_preds]  long
        mlm_weights  [B, max_num_mlm_preds]  float  (1 = real pred, 0 = padding)
        mlm_labels   [B, max_num_mlm_preds]  long
        nsp_labels   [B]                     long   (1 = is_next, 0 = not)
    """
    max_num_mlm_preds = round(max_len * 0.15)   # = 10 for max_len=64

    paragraphs = _read_wiki(csv_path)
    vocab      = Vocab(paragraphs)
    examples   = _get_nsp_data(paragraphs, vocab, max_len)
    tensors    = _pad_bert_inputs(examples, max_len, max_num_mlm_preds, vocab)

    loader = DataLoader(WikiDataset(tensors), batch_size=batch_size, shuffle=True)
    return loader, vocab


# ── quick sanity check ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader, vocab = load_data_wiki('dataset/wikitext2_paragraphs.csv')
    print(f'Vocab size   : {len(vocab):,}')
    print(f'Num batches  : {len(loader):,}')
    batch = next(iter(loader))
    names = ['token_ids', 'segments', 'valid_lens',
             'mlm_pos', 'mlm_weights', 'mlm_labels', 'nsp_labels']
    for name, t in zip(names, batch):
        print(f'  {name:14s}  shape={tuple(t.shape)}  dtype={t.dtype}')
