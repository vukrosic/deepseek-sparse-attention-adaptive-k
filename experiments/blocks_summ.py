"""Summarize blocks_sweep.sh runs for one text length: mean loss and tokens read per setting.  usage: python3 blocks_summ.py SEQ"""
import json, os, sys, torch
L = sys.argv[1]; os.environ["SEQ"] = L
import sampled as S
npos = {sk: S.gen(sk, 256, torch.Generator().manual_seed(1))[1].sum(1).float().mean().item() for sk in S.SKILLS}
wt = {sk: S.W[i].item() * npos[sk] for i, sk in enumerate(S.SKILLS)}; tot = sum(wt.values())
g = {}
for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampled_runs.jsonl")):
    r = json.loads(l)
    if r.get("seq") != L or not r.get("kept") or r.get("dsa_qt", "1") != "1": continue
    if r.get("dsa_blk") == "4" and "," in r.get("dsa_buckets", ""): key = f"chosen p={r['dsa_p']}"
    elif r.get("dsa_blk") == "4": key = f"fixed {int(r['dsa_buckets']):3d}"
    elif not r.get("dsa_p") and r.get("dsa_k") == L: key = "normal attention"
    else: continue
    g.setdefault(key, {})[r["seed"]] = r
for key, v in sorted(g.items()):
    v = list(v.values()); loss = sum(r["wloss"] for r in v) / len(v)
    read = sum(wt[s] * r["kept"][s] for r in v for s in S.SKILLS) / tot / len(v)
    print(f"L={L} {key:17s} n={len(v)} loss {loss:.3f}  read {read:6.1f}  seeds {[round(r['wloss'], 3) for r in v]}")
