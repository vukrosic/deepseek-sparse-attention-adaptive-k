"""Language-like toy: one stream of many skills, some frequent and some rare (Zipf), learned fully in ~5 s.
Every sequence is one skill, marked by a skill token at position 0 (like a document's genre), 16 positions.

skills, most to least frequent (weight 1/sqrt(rank))
  bigram   text from a fixed random Zipf chain   (word statistics)
  back2    copy the token 2 places back          (fixed-position attention)
  copy     a segment of random length repeats    (induction)
  recall   4 key-value pairs, then key -> value  (retrieval)
  sort     6 tokens, separator, sorted           (whole-context algorithm)
  reverse  6 tokens, separator, reversed
  max      8 tokens, separator, the largest
  count    6 tokens from {0..3}, how many 0s    (counting)
  fact     subject -> its fixed object, 48 facts (memorized knowledge, in the MLP)

usage: python3 mix.py MODE [seed] [steps] [lr]    MODE = whole | perhead | pow:A
prints per-skill loss every 50 steps and the frequency-weighted loss (like an LM's loss)
"""
import json, sys, time
import torch, torch.nn.functional as F
import suite
from suite import msign, H

SKILLS = ["bigram", "back2", "copy", "recall", "sort", "reverse", "max", "count", "fact"]
PAD, SEP = 64, 65
suite.V, suite.T = 66 + len(SKILLS), 16
L = 16
import os
ALPHA = float(os.environ.get("ALPHA", 0.5))               # skill frequency ~ 1 / rank^ALPHA
ADAM_LR = float(os.environ.get("ADAM_LR", 0.01))
W = torch.tensor([1 / (i + 1) ** ALPHA for i in range(len(SKILLS))]); W /= W.sum()
pos = torch.arange(L)
LOSS = None                                               # optional loss(model, x, y, msk, i, steps); default full cross-entropy
FACTS = torch.randint(64, (48,), generator=torch.Generator().manual_seed(5))


def gen(skill, n, g):
    """s: n x (L+1) token sequence (s[:,0] = skill token); m: n x L, True where y = s[:,1:] is scored"""
    r = lambda hi, *shape: torch.randint(hi, shape, generator=g)
    s = torch.full((n, L + 1), PAD); s[:, 0] = 66 + SKILLS.index(skill); m = torch.zeros(n, L, dtype=torch.bool)
    if skill == "bigram":
        s[:, 1] = r(64, n)
        for t in range(1, L): s[:, t + 1] = torch.multinomial(suite.CHAIN[s[:, t]], 1, generator=g)[:, 0]
        m[:, 1:] = True
    elif skill == "back2":
        s[:, 1:] = r(64, n, 2).repeat(1, L // 2); m[:, 2:] = True    # two random tokens repeating: s[t+1] = s[t-1]
    elif skill == "copy":                              # k random tokens then the same k again, k = 3..7
        k = r(5, n, 1) + 3; s[:, 1:] = r(64, n, L); p = pos.expand(n, L)
        src = torch.where((p >= k) & (p < 2 * k), p - k, p); s[:, 1:] = s[:, 1:].gather(1, src)
        m = (p > k) & (p < 2 * k)                      # y[t] = s[t+1] = content t; content k..2k-1 copies 0..k-1, the first is unguessable
    elif skill == "recall":                            # keys 0-15, values 16-31: k v k v k v k v k -> v
        ks = torch.rand(n, 16, generator=g).argsort(1)[:, :4]; vs = 16 + r(16, n, 4); j = r(4, n, 1)
        s[:, 1:9:2], s[:, 2:9:2] = ks, vs; s[:, 9] = ks.gather(1, j)[:, 0]; s[:, 10] = vs.gather(1, j)[:, 0]; m[:, 9] = True
    elif skill in ("sort", "reverse"):
        a = r(32, n, 6); s[:, 1:7] = a; s[:, 7] = SEP; s[:, 8:14] = a.sort(1).values if skill == "sort" else a.flip(1); m[:, 7:13] = True
    elif skill == "max":
        a = r(32, n, 8); s[:, 1:9] = a; s[:, 9] = SEP; s[:, 10] = a.max(1).values; m[:, 9] = True
    elif skill == "count":
        a = r(4, n, 6); s[:, 1:7] = a; s[:, 7] = SEP; s[:, 8] = (a == 0).sum(1); m[:, 7] = True
    elif skill == "fact":
        a = r(48, n); s[:, 1], s[:, 2], s[:, 3] = a, SEP, FACTS[a]; m[:, 2] = True
    return s, m


def batch(n, g):
    counts = torch.multinomial(W, n, replacement=True, generator=g).bincount(minlength=len(SKILLS))
    parts = [gen(sk, c.item(), g) for sk, c in zip(SKILLS, counts) if c > 0]
    s = torch.cat([p[0] for p in parts]); m = torch.cat([p[1] for p in parts])
    return s[:, :-1], s[:, 1:], m


def qk_steps(G, mode):
    """batched qk_step: G is n x out x in"""
    if mode == "whole": return msign(G)
    n, o, i = G.shape
    ph = msign(G.reshape(n * H, o // H, i)).reshape(n, o, i)
    if mode == "perhead": return ph
    U, S, Vh = torch.linalg.svd(ph, full_matrices=False)  # pow:a = per-head directions, singular values ^ a, same size
    d = U @ torch.diag_embed(S ** float(mode[4:])) @ Vh
    return d * (ph.norm(dim=(1, 2), keepdim=True) / d.norm(dim=(1, 2), keepdim=True))


QK_MAN, VO_MAN = os.environ.get("QK_MAN", "none"), os.environ.get("VO_MAN", "none")  # manifold for q,k / v,o after each Muon step
MAN_SCALE = float(os.environ.get("MAN_SCALE", 1))   # manifold size = MAN_SCALE x the matrix's init Frobenius norm
QK_MULT = float(os.environ.get("QK_MULT", 1))        # Muon lr multiplier for q,k only
LAYER_MULT = [float(v) for v in os.environ.get("LAYER_MULT", "1,1").split(",")]   # Muon lr multiplier per layer (layer 0, layer 1)
VO_MULT, MLP_MULT = float(os.environ.get("VO_MULT", 1)), float(os.environ.get("MLP_MULT", 1))         # Muon lr multipliers: v,o / MLP or experts
ROUTER_MULT, EMB_MULT = float(os.environ.get("ROUTER_MULT", 1)), float(os.environ.get("EMB_MULT", 1))  # AdamW lr multipliers: MoE router / embedding + output head
EMB_B1 = float(os.environ.get("EMB_B1", 0.9))          # Adam beta1 for embedding + head
HEAD_B1 = float(os.environ.get("HEAD_B1", EMB_B1))     # Adam beta1 for the head alone (default = EMB_B1)
REST_B1 = float(os.environ.get("REST_B1", 0.9))        # Adam beta1 for router + norm gains
ATT_MOM = float(os.environ.get("ATT_MOM", 0.95))       # Muon momentum for q,k,v,o
HEAD_MUON = float(os.environ.get("HEAD_MUON", 0))      # >0: output head on Muon at lr x HEAD_MUON (instead of AdamW)
HEAD_ROW = float(os.environ.get("HEAD_ROW", 0))        # >0: output head steps with momentum normalized per row (each word's vector moves RMS HEAD_ROW x ADAM_LR)
EXP_MOM = float(os.environ.get("EXP_MOM", 0.95))       # Muon momentum for MLP / expert matrices (attention keeps 0.95)
FREQ_LR = float(os.environ.get("FREQ_LR", 0))          # embedding row step x (count_i / mean count)^-FREQ_LR, clamped to [0.25, 4]
EMB_WARM, EMB_DECAY = int(os.environ.get("EMB_WARM", 30)), float(os.environ.get("EMB_DECAY", 0.5))  # embedding + head schedule: warmup steps, decay start fraction
LAZY = os.environ.get("LAZY", "0") == "1"
ADAM_B2 = float(os.environ.get("ADAM_B2", 0.98))         # Adam beta2 for every AdamW group except AdamW-on-matrices
HEAD_B2 = float(os.environ.get("HEAD_B2", ADAM_B2))         # Adam beta2 for the output head alone
MUON_ON = [k for k in os.environ.get("MUON_ON", "").split(",") if k]; ADAM_MAT_LR = float(os.environ.get("ADAM_MAT_LR", 0.005))
GF_L, GF_A, GF_ONLY = float(os.environ.get("GF_L", 0)), float(os.environ.get("GF_A", 0.98)), os.environ.get("GF_ONLY", "all"); gf = {}
NORM_MULT = float(os.environ.get("NORM_MULT", 1)); VE_MULT = float(os.environ.get("VE_MULT", os.environ.get("EMB_MULT", 1))); VE_B1 = float(os.environ.get("VE_B1", os.environ.get("EMB_B1", 0.9)))                                                       # AdamW lr multiplier: RMSNorm gains
HEAD_MULT = float(os.environ.get("HEAD_MULT", EMB_MULT))                                                # output head alone (default = EMB_MULT)
MLP_MAN = os.environ.get("MLP_MAN", "none")          # manifold for gate,up,down (sphere | stiefel; with MOE: per expert)
MLP_SCALE = float(os.environ.get("MLP_SCALE", 1))    # its size x init norm
QK_SCALE, VO_SCALE = (float(os.environ.get(k, MAN_SCALE)) for k in ("QK_SCALE", "VO_SCALE"))  # separate sizes for q,k and v,o
# none = free; sphere = whole matrix keeps its norm; stiefel = whole matrix has all singular values equal;
# sphere_head / stiefel_head = the same for each head's slice (rows of q,k,v; columns of o), every head the same size


def head_slices(p, nm):
    dh = p.shape[0] // H
    return [p[:, h * dh:(h + 1) * dh] if nm == "o" else p[h * dh:(h + 1) * dh] for h in range(H)]


def retract(p, nm, man, c):
    """put p back on its manifold of Frobenius size c (same size for every manifold, so only the shape differs)"""
    parts = head_slices(p, nm) if man.endswith("_head") else list(p.unbind(0)) if p.dim() == 3 else [p]   # 3D = experts: one manifold each
    cc = c / len(parts) ** 0.5
    for w in parts:
        if man.startswith("stiefel"): U, _, Vh = torch.linalg.svd(w, full_matrices=False); w.copy_(U @ Vh * cc / min(w.shape) ** 0.5)  # exact, even with MSIGN=ns
        else: w.copy_(w * cc / w.norm())


def run(mode, seed=0, steps=600, lr=0.03, every=50, bs=64, init=None):
    torch.manual_seed(seed); model = init if init is not None else suite.Model()  # init: continue training a given model
    names = [(l, n) for l in range(2) for n in suite.MATS]
    for pn, p in model.named_parameters():
        if "router" in pn: p.lr_mult = ROUTER_MULT
        elif pn.startswith(("ve.", "ke.")): p.lr_mult = VE_MULT; p.io = "ve"
        elif pn.startswith("emb"): p.lr_mult = EMB_MULT; p.io = "emb"
        elif pn.startswith("head"): p.lr_mult = HEAD_MULT; p.io = "head"
        elif pn == "hb": p.lr_mult = float(os.environ.get("HB_MULT", 1))
        elif ".cv." in pn or ".cv2." in pn: p.lr_mult = float(os.environ.get("CANON_MULT", 1))
        elif p.ndim == 1: p.lr_mult = NORM_MULT          # all RMSNorm gains (n1, n2, qn, kn, final norm)
    mats = [getattr(model.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    if HEAD_MUON or HEAD_ROW: ids.add(id(model.head.weight)); hbuf = torch.zeros_like(model.head.weight)
    if mode == "adam": mats, names = [], []                # baseline: AdamW on every parameter, matrices at lr, the rest at ADAM_LR
    amat = [p for nm, p in zip(names, mats) if MUON_ON and nm[1] not in MUON_ON]   # MUON_ON: only these matrix kinds use Muon, the rest AdamW at ADAM_MAT_LR
    if MUON_ON: names, mats = [nm for nm in names if nm[1] in MUON_ON], [p for nm, p in zip(names, mats) if nm[1] in MUON_ON]
    rest = [p for p in model.parameters() if id(p) not in ids]; mults = sorted({(getattr(p, "lr_mult", 1), getattr(p, "io", "")) for p in rest})   # lr_mult: e.g. Engram tables 5x
    adam = torch.optim.AdamW([dict(params=[p for p in rest if (getattr(p, "lr_mult", 1), getattr(p, "io", "")) == m], lr=ADAM_LR * m[0], io=m[1],
                                   betas=({"emb": EMB_B1, "head": HEAD_B1, "ve": VE_B1}.get(m[1], REST_B1), {"head": HEAD_B2}.get(m[1], ADAM_B2))) for m in mults]
                             + ([dict(params=[p for p in model.parameters() if id(p) in ids], lr=lr)] if mode == "adam" else [])
                             + ([dict(params=amat, lr=ADAM_MAT_LR)] if amat else []), betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]
    man = {nm: QK_MAN if nm[1] in "qk" else VO_MAN if nm[1] in "vo" else MLP_MAN for nm in names}
    with torch.no_grad():
        size = {nm: (QK_SCALE if nm[1] in "qk" else VO_SCALE if nm[1] in "vo" else MLP_SCALE) * p.norm().item() for nm, p in zip(names, mats)}
        for nm, p in zip(names, mats):
            if man.get(nm, "none") != "none": retract(p, nm[1], man[nm], size[nm])
    g, ge = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(int(os.environ.get("EVAL_SEED", 999)))
    ev = {sk: gen(sk, 128, ge) for sk in SKILLS}
    rows = []
    for i in range(steps + 1):
        if i % every == 0:
            with torch.no_grad():
                row = {}
                for sk, (s, m) in ev.items():
                    out = model(s[:, :-1]); y = s[:, 1:]
                    row[sk] = (round(F.cross_entropy(out[m], y[m]).item(), 3), round((out.argmax(-1)[m] == y[m]).float().mean().item(), 2))
                rows.append((i, row))
        if i == steps: break
        x, y, msk = batch(bs, g)
        loss = LOSS(model, x, y, msk, i, steps) if LOSS else F.cross_entropy(model(x)[msk], y[msk])
        if suite.DSA_K: loss = loss + sum(l.aux for l in model.layers if l.aux is not None)   # indexer KL; touches only the indexer (its inputs and targets are detached)
        f = suite.lr_at(i, steps)
        fio = min(1.0, (i + 1) / max(1, EMB_WARM)) * min(1.0, (1 - i / steps) / (1 - EMB_DECAY)) if suite.SCHED else 1.0
        for gr in adam.param_groups: gr["lr"] = gr.setdefault("base", gr["lr"]) * (fio if gr.get("io") else f)
        if FREQ_LR:
            cnt = cnt + torch.bincount(x.flatten(), minlength=model.emb.weight.shape[0]) if i else torch.bincount(x.flatten(), minlength=model.emb.weight.shape[0]).float()
            e0 = model.emb.weight.detach().clone()
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward()
        if GF_L:                                                                 # Grokfast (arXiv 2405.20233): g += lambda * EMA(g)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None or (GF_ONLY == "mats" and not any(p is m_ for m_ in mats)): continue
                    e = gf.setdefault(p, torch.zeros_like(p)); e.mul_(GF_A).add_(p.grad, alpha=1 - GF_A); p.grad.add_(e, alpha=GF_L)
        if LAZY:                                                                 # lazy Adam on input tables: rows with no gradient stay put
            lz = [(p, p.grad.abs().sum(1) == 0, p.detach().clone(), {k: v.clone() for k, v in adam.state[p].items() if torch.is_tensor(v) and v.shape == p.shape})
                  for gr in adam.param_groups if gr.get("io") in ("emb", "ve") for p in gr["params"] if p.grad is not None and p.ndim == 2]
        adam.step()
        if LAZY:
            with torch.no_grad():
                for p, z, old, st in lz:
                    p[z] = old[z]
                    for k, v in st.items(): adam.state[p][k][z] = v[z]
        if FREQ_LR:
            with torch.no_grad():
                sc = ((cnt + 1) / (cnt + 1).mean()).pow(-FREQ_LR).clamp(0.25, 4)[:, None]
                model.emb.weight.copy_(e0 + (model.emb.weight - e0) * sc)
        if not mats: continue
        with torch.no_grad():
            if HEAD_MUON:
                hw = model.head.weight; hbuf.mul_(0.95).add_(hw.grad)
                hw.add_(msign(hw.grad + 0.95 * hbuf), alpha=-lr * f * HEAD_MUON * max(1, hw.shape[0] / hw.shape[1]) ** 0.5); hw.grad = None
            if HEAD_ROW:
                hw = model.head.weight; hbuf.mul_(EMB_B1).add_(hw.grad, alpha=1 - EMB_B1); uu = hbuf
                hw.add_(uu / (uu.pow(2).mean(1, keepdim=True).sqrt() + 1e-12), alpha=-ADAM_LR * HEAD_ROW * (fio if EMB_WARM != 30 or EMB_DECAY != 0.5 else f)); hw.grad = None
            mom = [ATT_MOM if nm[1] in ("q", "k", "v", "o", "ag") else EXP_MOM for nm in names]
            for b, p, mo in zip(buf, mats, mom): b.mul_(mo).add_(p.grad)
            u = {nm: p.grad + mo * b for nm, p, b, mo in zip(names, mats, buf, mom)}
            st = {}
            for kinds in (("q", "k"), ("v", "o"), ("gate", "up"), ("down",)) + ((("ag",),) if suite.ATTN_GATE else ()) + ((("iq",), ("ik",)) if "iq" in suite.MATS else ()):  # one batched SVD per group of same-shape matrices
                keys = [nm for nm in names if nm[1] in kinds]
                if not keys: continue
                G = torch.stack([u[nm] for nm in keys])
                S = qk_steps(G, mode) if kinds == ("q", "k") else msign(G)
                st.update(zip(keys, S))
            for nm, p in zip(names, mats):
                p.add_(st[nm], alpha=-lr * f * max(1, p.shape[-2] / p.shape[-1]) ** 0.5 * (QK_MULT if nm[1] in "qk" else VO_MULT if nm[1] in "vo" else MLP_MULT) * LAYER_MULT[nm[0]])
                if man.get(nm, "none") != "none": retract(p, nm[1], man[nm], size[nm])
    global last_model; last_model = model
    return rows


def weighted(row): return sum(w.item() * row[sk][0] for w, sk in zip(W, SKILLS))


if __name__ == "__main__":
    mode = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
    t0 = time.time(); rows = run(mode, seed, steps, lr)
    print(f"{mode} seed {seed} lr {lr}  {time.time() - t0:.1f}s   accuracy per skill (loss weighted by frequency)")
    print("step  " + " ".join(f"{sk:>7}" for sk in SKILLS) + "  wloss")
    for i, row in rows: print(f"{i:4}  " + " ".join(f"{row[sk][1]:7.2f}" for sk in SKILLS) + f"  {weighted(row):.3f}")
    with open("mix_runs.jsonl", "a") as f: f.write(json.dumps(dict(mode=mode, seed=seed, steps=steps, rows=rows)) + "\n")
