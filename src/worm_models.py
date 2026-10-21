"""Phase 1 — shared-core worm models with differently indexed readouts.

One architecture; only the KEY that indexes the per-neuron tables changes:

    M0  key = (animal, column)             per-animal readout (MICrONS recipe); uses unnamed columns
    M1  key = neuron name                  shared across animals; unnamed columns ignored
    M2  key = name, permuted per animal     M1 with names shuffled independently in each animal
    level="class" collapses names to classes (AVAL, AVAR -> AVA; IL2DL -> IL2) for M1/M2.

Per recording, with observed activity x_t (N_rec) and mask m_t:
    e_t   = sum_i m_i x_i enc[key_i] / (sum_i m_i + 1)          encoder, (d,)
    s_t   = sum_j u_t[j] stim[key_j]                            stimulation input, (d,)
    h_t   = GRU([e_t, s_t], h_{t-1})                            shared core
    x^_{t+1}[i] = dec[key_i] . [h_t, 1]                         readout

Training crops: PRE=30 steps with observations (teacher forcing), then POST=60 steps
free-running (the model's own predictions replace observations as encoder input),
stimulation applied at its true frame. Crops are centred on stimulation events, plus an
equal number of random crops. Loss: masked MSE on all predicted steps.

Evaluation of an event (animal a, frame t, target j): burn in on the real trace over
[t-30, t), then free-run 60 steps twice, with and without the stimulus; predicted response
= integrated difference (counterfactual effect of the stimulus). Pooled over events into a
STAM per (responder name, target name) and correlated with the measured STAM exactly as in
Creamer et al. (metrics.nan_corr, diagonal masked).
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import torch
import torch.nn as nn

PRE, POST = 30, 60

_MULTI = ("IL1", "IL2", "CEP", "OLQ", "URA", "URY", "RMD", "RME", "SAA", "SMD", "SMB",
          "SIA", "SIB", "SAB", "OLL", "URB", "GLR", "IL2D", "IL2V", "IL1D", "IL1V")


def neuron_class(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r"[LR]$", "", name) if len(name) > 2 else name
    for base in sorted(_MULTI, key=len, reverse=True):
        if n.startswith(base) and len(n) - len(base) <= 1:
            return base[:3] if base[:3] in _MULTI else base
    return n


@dataclasses.dataclass
class Rec:
    name: str
    x: np.ndarray        # (T, N) float32, NaN = unobserved
    u: np.ndarray        # (T, N) int8 stimulation
    ids: np.ndarray      # (N,) names ("" = unnamed)
    keys: np.ndarray | None = None  # (N,) table index, -1 = excluded


def assign_keys(recs: list[Rec], scheme: str, level: str, vocab: dict | None, rng: np.random.Generator,
                m0_offset: int = 0) -> tuple[dict, int]:
    """Fill rec.keys. Returns (vocab, table size). For M0 keys are fresh per (animal, column)."""
    if scheme == "M0":
        k = m0_offset
        for r in recs:
            r.keys = np.arange(k, k + len(r.ids))
            k += len(r.ids)
        return {}, k
    labels = [[neuron_class(s) if level == "class" else s for s in r.ids] for r in recs]
    if vocab is None:
        vocab = {s: i for i, s in enumerate(sorted({s for L in labels for s in L if s}))}
    for r, L in zip(recs, labels):
        keys = np.array([vocab.get(s, -1) if s else -1 for s in L])
        if scheme == "M2":
            named = np.where(keys >= 0)[0]
            keys[named] = keys[named][rng.permutation(len(named))]
        r.keys = keys
    return vocab, len(vocab)


class LinearCell(nn.Module):
    """Drop-in for nn.GRUCell with linear dynamics: h' = A h + B x + b (A initialised to 0.5 I)."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.A = nn.Linear(hidden, hidden, bias=False)
        self.B = nn.Linear(n_in, hidden)
        with torch.no_grad():
            self.A.weight.copy_(0.5 * torch.eye(hidden))

    def forward(self, x, h):
        return self.A(h) + self.B(x)


class DiagLinearCell(nn.Module):
    """Stable linear core: h' = diag(0.99 * sigmoid(a)) h + B x (eigenvalues in (0, 0.99))."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.a = nn.Parameter(torch.linspace(-2.0, 3.0, hidden))
        self.B = nn.Linear(n_in, hidden)

    def forward(self, x, h):
        return 0.99 * torch.sigmoid(self.a) * h + self.B(x)


class SNLinearCell(nn.Module):
    """Stable full linear core: h' = 0.99 * A/||A||_2 h + B x (spectral norm; allows rotations/oscillations)."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.A = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, hidden, bias=False))
        self.B = nn.Linear(n_in, hidden)

    def forward(self, x, h):
        return 0.99 * self.A(h) + self.B(x)


class WormNet(nn.Module):
    openloop = False  # set per instance; True = latent driven only by inputs after the burn-in
    """tie=True: no stim table; the stimulation vector of column j is stim_map(enc[key_j]), so a
    neuron's input effect is a shared function of its (activity-learned) encoder row. This lets a
    per-animal readout (M0) express the response to stimulating a column it never saw stimulated.
    rank>0: each per-key table row is a rank-r coefficient vector times a shared basis
    (parameter-matched per-animal readouts)."""

    def __init__(self, n_keys: int, d: int = 32, hidden: int = 64, tie: bool = False, rank: int = 0,
                 core: str = "gru"):
        super().__init__()
        self.tie, self.rank, self.d = tie, rank, d
        self.enc = nn.Embedding(n_keys, rank or d)
        if not tie:
            self.stim = nn.Embedding(n_keys, rank or d)
        self.dec = nn.Embedding(n_keys, rank or hidden + 1)
        for name in self.row_names():
            nn.init.normal_(getattr(self, name).weight, std=0.1 if not rank else 1.0)
        if rank:
            self.B_enc = nn.Parameter(torch.randn(rank, d) * 0.1 / rank**0.5)
            self.B_dec = nn.Parameter(torch.randn(rank, hidden + 1) * 0.1 / rank**0.5)
            if not tie:
                self.B_stim = nn.Parameter(torch.randn(rank, d) * 0.1 / rank**0.5)
        if tie:
            self.stim_map = nn.Linear(d, d, bias=False)
        self.gru = {"gru": nn.GRUCell, "linear": LinearCell, "lineardiag": DiagLinearCell,
                    "linearsn": SNLinearCell}[core](2 * d, hidden)
        self.hidden = hidden

    def row_names(self) -> list[str]:
        return ["enc", "dec"] + ([] if self.tie else ["stim"])

    def lookup(self, k):
        enc, dec = self.enc(k), self.dec(k)
        if self.rank:
            enc, dec = enc @ self.B_enc, dec @ self.B_dec
        if self.tie:
            stim = self.stim_map(enc)
        else:
            stim = self.stim(k) @ self.B_stim if self.rank else self.stim(k)
        return enc, stim, dec

    def grow(self, n_new: int, init: str = "random") -> None:
        """Append rows (for few-shot on new animals). init='popmean' starts new rows at the mean of
        the existing rows (an identity-free population prior)."""
        for name in self.row_names():
            old = getattr(self, name)
            new = nn.Embedding(old.num_embeddings + n_new, old.embedding_dim).to(old.weight.device)
            with torch.no_grad():
                new.weight[: old.num_embeddings] = old.weight
                if init == "popmean":
                    new.weight[old.num_embeddings :] = old.weight.mean(0, keepdim=True)
                else:
                    nn.init.normal_(new.weight[old.num_embeddings :], std=0.1 if not self.rank else 1.0)
            setattr(self, name, new)

    def rollout(self, x, m, u, keys, n_teacher: int, stim_on: bool = True, inj_x=None, inj_m=None):
        """x, m, u: (B, T, N); keys: (B, N) with -1 = excluded (masked out everywhere).
        Teacher-forced for n_teacher steps, then free-running. Returns predictions (B, T-1, N)
        of x[:, 1:]."""
        B, T, N = x.shape
        valid = keys >= 0
        k = keys.clamp(min=0)
        enc, stim, dec = self.lookup(k)                                  # (B,N,d),(B,N,d),(B,N,H+1)
        enc = enc * valid[..., None]
        stim = stim * valid[..., None]
        dev = x.device
        h = torch.zeros(B, self.hidden, device=dev)
        one = torch.ones(B, 1, device=dev)
        cur_x, cur_m = x[:, 0], m[:, 0]
        preds = []
        for t in range(T - 1):
            e = torch.einsum("bn,bnd->bd", cur_x * cur_m, enc) / (cur_m.sum(1, keepdim=True) + 1)
            s = torch.einsum("bn,bnd->bd", u[:, t] if stim_on else torch.zeros_like(u[:, t]), stim)
            h = self.gru(torch.cat([e, s], 1), h)
            y = torch.einsum("bh,bnh->bn", torch.cat([h, one], 1), dec)
            preds.append(y)
            if t + 1 < n_teacher:
                cur_x, cur_m = x[:, t + 1], m[:, t + 1]
            elif self.openloop:  # open loop: no observation and no prediction feedback after burn-in
                if inj_x is not None:  # activity injection ("stimulate j" for models trained without stimulation)
                    cur_x, cur_m = inj_x[:, t + 1], inj_m[:, t + 1]
                else:
                    cur_x, cur_m = torch.zeros_like(y), torch.zeros_like(y)
            else:  # free-run: feed own prediction, for every column the encoder can see
                cur_x, cur_m = y, valid.float()
        return torch.stack(preds, 1)


DEVICE = "cpu"  # set by the driver; e.g. "cuda"


def crop_batch(recs: list[Rec], items: list[tuple[int, int]], max_n: int):
    """items: (rec index, crop start). Pads columns to max_n."""
    B, T = len(items), PRE + POST
    x = torch.zeros(B, T, max_n)
    m = torch.zeros(B, T, max_n)
    u = torch.zeros(B, T, max_n)
    keys = torch.full((B, max_n), -1, dtype=torch.long)
    for b, (ri, s) in enumerate(items):
        r = recs[ri]
        n = r.x.shape[1]
        seg = r.x[s : s + T]
        x[b, :, :n] = torch.from_numpy(np.nan_to_num(seg))
        m[b, :, :n] = torch.from_numpy(~np.isnan(seg)).float()
        u[b, :, :n] = torch.from_numpy(r.u[s : s + T].astype(np.float32))
        keys[b, :n] = torch.from_numpy(r.keys)
    return x.to(DEVICE), m.to(DEVICE), u.to(DEVICE), keys.to(DEVICE)


def event_list(recs: list[Rec], idx: list[int] | None = None, frame_range=None) -> list[tuple[int, int, int]]:
    """(rec index, frame, target column) for stimulation events with a full window."""
    out = []
    for ri in (idx if idx is not None else range(len(recs))):
        r = recs[ri]
        for t, j in zip(*np.where(r.u == 1)):
            if t - PRE >= 0 and t + POST < r.x.shape[0]:
                if frame_range is None or (frame_range[ri][0] <= t < frame_range[ri][1]):
                    out.append((ri, int(t), int(j)))
    return out


def train(model: WormNet, recs: list[Rec], steps: int, rng: np.random.Generator, batch: int = 32,
          lr: float = 3e-3, params=None, events=None, log_every: int = 500, tag: str = "",
          reg=None, resp_loss: float = 0.0, resp_blind: bool = False, bg_model=None, bg_alpha: float = 0.0, bg_keymap=None) -> list[float]:
    events = events if events is not None else event_list(recs)
    max_n = max(r.x.shape[1] for r in recs)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        items = []
        for _ in range(batch // 2 if events else 0):
            ri, t, _ = events[rng.integers(len(events))]
            items.append((ri, t - PRE))
        for _ in range(batch - len(items)):
            ri = rng.integers(len(recs))
            items.append((ri, int(rng.integers(0, recs[ri].x.shape[0] - PRE - POST))))
        x, m, u, keys = crop_batch(recs, items, max_n)
        pred = model.rollout(x, m, u, keys, n_teacher=PRE)
        tgt, tm = x[:, 1:], m[:, 1:] * (keys >= 0)[:, None, :]
        loss = (((pred - tgt) ** 2) * tm).sum() / tm.sum().clamp(min=1)
        ne = (batch // 2) if events else 0
        if resp_loss > 0 and ne:   # event-level counterfactual response loss (the atoms of the STAM metric)
            xe, me, ue, ke = x[:ne], m[:ne], u[:ne], keys[:ne]
            if resp_blind:   # state-free: the effect must be predicted without the pre-stimulus observations
                xb, mb = xe.clone(), me.clone(); xb[:, :PRE] = 0; mb[:, :PRE] = 0
                on = model.rollout(xb, mb, ue, ke, n_teacher=PRE, stim_on=True)
                off = model.rollout(xb, mb, ue, ke, n_teacher=PRE, stim_on=False)
                eff = (on - off)[:, PRE - 1:].sum(1)
            else:
                off = model.rollout(xe, me, ue, ke, n_teacher=PRE, stim_on=False)
                eff = (pred[:ne] - off)[:, PRE - 1:].sum(1)
            post_m, pre_m = me[:, PRE:].sum(1), me[:, :PRE].sum(1)
            meas = (xe[:, PRE:] * me[:, PRE:]).sum(1) / post_m.clamp(min=1) - (xe[:, :PRE] * me[:, :PRE]).sum(1) / pre_m.clamp(min=1)
            meas = meas * POST
            if bg_model is not None and bg_alpha:   # residual target: subtract the frozen background (no-stimulation) forecast
                with torch.no_grad():
                    kb = torch.where(ke >= 0, bg_keymap[ke.clamp(min=0)], ke) if bg_keymap is not None else ke
                    bo = bg_model.rollout(xe, me, ue, kb, n_teacher=PRE, stim_on=False)
                    bchg = (bo[:, PRE - 1:PRE - 1 + POST].mean(1) - (xe[:, :PRE] * me[:, :PRE]).sum(1) / pre_m.clamp(min=1)) * POST
                meas = meas - bg_alpha * bchg
            ok = (post_m > POST // 2) & (pre_m > PRE // 2) & (ke >= 0) & (ue[:, PRE] == 0)   # exclude the stimulated column
            loss = loss + resp_loss * (((eff - meas) ** 2) * ok).sum() / ok.sum().clamp(min=1) / POST
        if reg is not None:
            loss = loss + reg()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            print(f"    {tag} step {step + 1}/{steps}  loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return losses


@torch.no_grad()
def predict_events(model: WormNet, recs: list[Rec], events, chunk: int = 64):
    """Counterfactual integrated response for each event: (len(events), N_rec) arrays (ragged)."""
    max_n = max(r.x.shape[1] for r in recs)
    out = []
    for i in range(0, len(events), chunk):
        ev = events[i : i + chunk]
        x, m, u, keys = crop_batch(recs, [(ri, t - PRE) for ri, t, _ in ev], max_n)
        on = model.rollout(x, m, u, keys, n_teacher=PRE, stim_on=True)
        off = model.rollout(x, m, u, keys, n_teacher=PRE, stim_on=False)
        eff = (on - off)[:, PRE - 1 :].sum(1)   # predictions of frames PRE..PRE+POST-1
        for b, (ri, _, _) in enumerate(ev):
            out.append(eff[b, : recs[ri].x.shape[1]].cpu().numpy())
    return out


def pooled_stam(recs: list[Rec], events, responses, names: list[str]) -> np.ndarray:
    """Average per-event responses into a (responder, target) matrix on the `names` axis."""
    col = {c: k for k, c in enumerate(names)}
    acc = np.zeros((len(names), len(names)))
    cnt = np.zeros_like(acc)
    for (ri, _, j), resp in zip(events, responses):
        r = recs[ri]
        tj = col.get(r.ids[j])
        if tj is None or not r.ids[j]:
            continue
        for i, nm in enumerate(r.ids):
            ci = col.get(nm)
            if ci is not None and nm and np.isfinite(resp[i]):
                acc[ci, tj] += resp[i]
                cnt[ci, tj] += 1
    s = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    s[np.eye(len(names), dtype=bool)] = np.nan
    return s


def measured_event_responses(recs: list[Rec], events):
    out = []
    for ri, t, _ in events:
        r = recs[ri]
        clip = r.x[t - PRE : t + POST]
        with np.errstate(all="ignore"):
            base = np.nanmean(clip[:PRE], axis=0)
            v = np.nansum(clip[PRE:] - base, axis=0)
        v[np.all(np.isnan(clip[PRE:]), axis=0)] = np.nan
        out.append(v)
    return out


class ExprNet(WormNet):
    """Phase 3 (M3/M4): no free per-key rows. Every key's enc/stim/dec row is a learned function
    of a fixed expression feature vector G[key] (CeNGEN; M4 = class profiles shuffled across classes), so
    neurons with the same expression share rows, and a neuron never seen in training still gets rows."""

    def __init__(self, G: np.ndarray, d: int = 32, hidden: int = 64, width: int = 64):
        nn.Module.__init__(self)
        self.tie, self.rank, self.d, self.hidden = False, 0, d, hidden
        self.register_buffer("G", torch.as_tensor(G, dtype=torch.float32))
        f = G.shape[1]
        mlp = lambda out: nn.Sequential(nn.Linear(f, width), nn.GELU(), nn.Linear(width, out))
        self.f_enc, self.f_stim, self.f_dec = mlp(d), mlp(d), mlp(hidden + 1)
        self.gru = nn.GRUCell(2 * d, hidden)

    def row_names(self) -> list[str]:
        return []

    def lookup(self, k):
        g = self.G[k]
        return self.f_enc(g), self.f_stim(g), self.f_dec(g)
