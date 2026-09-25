# SPDX-License-Identifier: MIT
"""Data, model, beam search and metrics for the Multi30k experiment (see README.md)."""

from __future__ import annotations

import json
import math
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import beamgrad

PAD, BOS, EOS, UNK = 0, 1, 2, 3
DATA_URL = "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/{split}.jsonl"
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # test = test2016 (flickr)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{SPLITS[split]}.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(split=SPLITS[split]), path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_tokenizer(texts: list[str], vocab_size: int, path: Path):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )
    tokenizer.train_from_iterator(texts, trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def encode_pairs(tokenizer, rows: list[dict], src: str = "en", tgt: str = "de", max_len: int = 64):
    """Token-id lists (source; target without BOS/EOS), truncated to max_len."""
    sources = [e.ids[:max_len] for e in tokenizer.encode_batch([r[src] for r in rows])]
    targets = [e.ids[: max_len - 1] for e in tokenizer.encode_batch([r[tgt] for r in rows])]
    return list(zip(sources, targets, strict=True))


def pad(sequences: list[list[int]], value: int, device) -> torch.Tensor:
    width = max(len(s) for s in sequences)
    out = torch.full((len(sequences), width), value, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out.to(device)


class Batch:
    """A batch: source ids and mask, and the reference (target + EOS)."""

    def __init__(self, pairs, device):
        self.size = len(pairs)
        self.src = pad([s for s, _ in pairs], PAD, device)
        self.src_mask = self.src != PAD
        self.gold = pad([t + [EOS] for _, t in pairs], -1, device)  # [B, L], -1 after EOS
        self.gold_lengths = (self.gold >= 0).sum(1)
        self.tgt_in = pad([[BOS] + t for _, t in pairs], PAD, device)  # teacher-forcing inputs
        self.references = [t for _, t in pairs]


def batches(pairs, batch_size: int, device, generator: torch.Generator | None = None):
    order = torch.randperm(len(pairs), generator=generator).tolist() if generator is not None else range(len(pairs))
    order = list(order)
    for i in range(0, len(order), batch_size):
        yield Batch([pairs[j] for j in order[i : i + batch_size]], device)


# ---------------------------------------------------------------------------
# Model: a pre-LN transformer with a key/value cache that follows the beams
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def keys_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, S, D = x.shape
        k, v = self.kv(x).view(N, S, 2, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        return k, v  # [N, h, S, dh]

    def forward(self, x, k, v, mask=None, causal=False):
        N, T, D = x.shape
        q = self.q(x).view(N, T, self.heads, D // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
        )
        return self.o(y.transpose(1, 2).reshape(N, T, D))


class FeedForward(nn.Sequential):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, hidden, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.ff = FeedForward(d, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, *self.attn.keys_values(h), mask=mask))
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, hidden, dropout):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, heads, dropout)
        self.cross = Attention(d, heads, dropout)
        self.ff = FeedForward(d, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross_kv, memory_mask, past=None):
        """x: [N, t, D] new positions; past: (k, v) of the positions before them, or None."""
        h = self.norm1(x)
        k, v = self.self_attn.keys_values(h)
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        # Teacher forcing (no past) is causal; one new position after a past attends to all of it.
        x = x + self.drop(self.self_attn(h, k, v, causal=past is None and x.shape[1] > 1))
        x = x + self.drop(self.cross(self.norm2(x), *cross_kv, mask=memory_mask))
        return x + self.drop(self.ff(self.norm3(x))), (k, v)


class Transformer(nn.Module):
    def __init__(self, vocab, d=256, heads=4, layers=3, hidden=1024, dropout=0.1, max_positions=256):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d, padding_idx=PAD)  # shared by source, target and output
        self.positions = nn.Embedding(max_positions, d)
        self.encoder = nn.ModuleList([EncoderLayer(d, heads, hidden, dropout) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d, heads, hidden, dropout) for _ in range(layers)])
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, std=d**-0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def set_dropout(self, p: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
            elif isinstance(m, Attention):
                m.dropout = p

    def _embed(self, tokens, offset=0):
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        return self.drop(self.embed(tokens) * math.sqrt(self.d) + self.positions(positions))

    def encode(self, src, src_mask):
        """Memory [B, S, D] and the cross-attention mask [B, 1, 1, S]."""
        mask = src_mask[:, None, None, :]
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def cross_kv(self, memory):
        return [layer.cross.keys_values(memory) for layer in self.decoder]

    def decode(self, tokens, cross, memory_mask, past=None, offset=0):
        """Logits [N, t, V] of t new positions, and the key/value cache after them."""
        x = self._embed(tokens, offset)
        cache = []
        for i, layer in enumerate(self.decoder):
            x, kv = layer(x, cross[i], memory_mask, None if past is None else past[i])
            cache.append(kv)
        return self.dec_norm(x) @ self.embed.weight.T, cache

    def token_log_probs(self, src, src_mask, tgt_in, targets):
        """Teacher-forced log-probabilities [N, T] of targets (-1 = padding, gets 0)."""
        memory, mask = self.encode(src, src_mask)
        logits, _ = self.decode(tgt_in, self.cross_kv(memory), mask)
        lp = logits.float().log_softmax(-1)
        out = lp.gather(-1, targets.clamp(min=0)[..., None])[..., 0]
        return out.masked_fill(targets < 0, 0.0), lp


class BeamStep:
    """beamgrad step function: next-token log-probs of every beam, [B, K, V].

    The encoder runs once; the decoder runs one position per step, with a
    key/value cache reordered by each beam's parent slot.
    """

    def __init__(self, model: Transformer, src, src_mask, beam_size: int):
        self.model, self.B, self.K = model, src.shape[0], beam_size
        memory, mask = model.encode(src, src_mask)
        self.mask = mask.repeat_interleave(beam_size, 0)
        self.cross = [
            (k.repeat_interleave(beam_size, 0), v.repeat_interleave(beam_size, 0)) for k, v in model.cross_kv(memory)
        ]
        self.cache = None

    def __call__(self, beams: beamgrad.BeamState) -> torch.Tensor:
        B, K = self.B, self.K
        if beams.step == 0:
            tokens = torch.full((B * K, 1), BOS, dtype=torch.long, device=self.mask.device)
        else:
            index = (torch.arange(B, device=beams.parents.device)[:, None] * K + beams.parents.clamp(min=0)).flatten()
            self.cache = [(k[index], v[index]) for k, v in self.cache]
            tokens = beams.tokens.clamp(min=0).reshape(B * K, 1)
        logits, self.cache = self.model.decode(tokens, self.cross, self.mask, self.cache, offset=beams.step)
        return logits[:, -1].float().log_softmax(-1).view(B, K, -1)


class Rescorer:
    """beamgrad rescore_fn: teacher-forced token log-probs [B, N, T] of whole sequences."""

    def __init__(self, model: Transformer, src, src_mask):
        self.model, self.src, self.src_mask = model, src, src_mask

    def __call__(self, sequences, lengths):
        B, N, T = sequences.shape
        seqs = sequences.reshape(B * N, T)
        targets = torch.where(torch.arange(T, device=seqs.device) < lengths.reshape(-1, 1), seqs, -1)
        tgt_in = torch.cat([torch.full_like(seqs[:, :1], BOS), seqs[:, :-1].clamp(min=0)], dim=1)
        src = self.src.repeat_interleave(N, 0)
        lp, _ = self.model.token_log_probs(src, self.src_mask.repeat_interleave(N, 0), tgt_in, targets)
        return lp.view(B, N, T)


def max_steps(batch: Batch) -> int:
    return min(int(batch.src_mask.sum(1).max()) * 3 // 2 + 5, 100)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def sentence_bleu(hypothesis: list[int], reference: list[int]) -> float:
    """Smoothed sentence BLEU (add-one for n > 1, Lin & Och 2004) on token ids."""
    if not hypothesis:
        return 0.0
    log_precision = 0.0
    for n in range(1, 5):
        hyp = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        ref = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        match = sum(min(c, ref[g]) for g, c in hyp.items())
        total = max(len(hypothesis) - n + 1, 0)
        if n == 1:
            if match == 0:
                return 0.0
            log_precision += math.log(match / total) / 4
        else:
            log_precision += math.log((match + 1) / (total + 1)) / 4
    brevity = min(0.0, 1.0 - len(reference) / len(hypothesis))
    return math.exp(log_precision + brevity)


def strip(sequence: list[int]) -> list[int]:
    """Token ids of a hypothesis up to (not including) EOS / padding."""
    out = []
    for t in sequence:
        if t < 0 or t == EOS:
            break
        out.append(t)
    return out


@torch.no_grad()
def translate(model, pairs, device, beam_size=4, alpha=0.6, batch_size=128) -> list[list[int]]:
    was_training = model.training
    model.eval()
    options = beamgrad.BeamOptions(
        beam_size=beam_size, eos_token=EOS, length_penalty_alpha=alpha, banned_tokens=(PAD, BOS, UNK)
    )
    out = []
    for batch in batches(pairs, batch_size, device):
        step = BeamStep(model, batch.src, batch.src_mask, beam_size)
        result = beamgrad.beam_search(step, options, max_steps(batch), batch_size=batch.size)
        out.extend(strip(s) for s in result.sequences[:, 0].tolist())
    model.train(was_training)
    return out


def corpus_bleu(tokenizer, hypotheses: list[list[int]], references: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(tokenizer.decode_batch(hypotheses), [references]).score
