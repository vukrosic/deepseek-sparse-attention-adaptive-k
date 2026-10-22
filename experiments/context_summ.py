"""Summarize context_sweep.sh runs for one sequence length: mean loss and tokens read per setting.  usage: python3 context_summ.py SEQ"""
import json, os, sys, torch
L = sys.argv[1]; os.environ["SEQ"] = L
import sampled as S
npos = {sk: S.gen(sk, 256, torch.Generator().manual_seed(1))[1].sum(1).float().mean().item() for sk in S.SKILLS}
wt = {sk: S.W[i].item() * npos[sk] for i, sk in enumerate(S.SKILLS)}; tot = sum(wt.values())
g = {}
for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampled_runs.jsonl")):
    r = json.loads(l)
    if r.get("seq") != L or not r.get("kept") or r.get("dsa_blk", "1") != "1" or r.get("dsa_qt", "1") != "1" or r.get("dsa_rule", "p") != "p": continue
    if r.get("dsa_p") == "0.5" and r.get("dsa_buckets", "").split(",")[-1] == L: key = ("min", int(r["dsa_buckets"].split(",")[0]))
    elif not r.get("dsa_p") and r.get("dsa_k") != L: key = ("fixed", int(r["dsa_k"]))
    else: continue
    g.setdefault(key, {})[r["seed"]] = r
out = {}
for key, v in sorted(g.items()):
    v = list(v.values()); loss = sum(r["wloss"] for r in v) / len(v)
    read = sum(wt[s] * r["kept"][s] for r in v for s in S.SKILLS) / tot / len(v)
    out[f"{key[0]}{key[1]}"] = dict(n=len(v), loss=round(loss, 3), read=round(read, 1), seeds=[round(r["wloss"], 3) for r in v])
    print(f"L={L} {key[0]:5s} {key[1]:3d}  n={len(v)} loss {loss:.3f}  read {read:5.1f}  seeds {out[f'{key[0]}{key[1]}']['seeds']}")
