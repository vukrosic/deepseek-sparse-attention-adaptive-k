"""Group the runs in sampled_runs.jsonl by setting: mean loss and tokens kept per word (each task weighted by its share of scored words). usage: python3 summ.py"""
import json, os, sys, torch
os.environ["SEQ"] = "64"
import sampled as S
npos = {sk: S.gen(sk, 256, torch.Generator().manual_seed(1))[1].sum(1).float().mean().item() for sk in S.SKILLS}
wt = {sk: S.W[i].item() * npos[sk] for i, sk in enumerate(S.SKILLS)}; tot = sum(wt.values())
g = {}
for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampled_runs.jsonl")):
    r = json.loads(l)
    if r.get("seq") != "64" or not r.get("kept") or r.get("dsa_kc") != "1" or r.get("dsa_muon") != "1" or "dsa_train" in r: continue
    key = (r.get("dsa_rule", "p"), r.get("dsa_p") or f"k{r['dsa_k']}", (r.get("dsa_buckets") or "4..64")[:14], r.get("dsa_blk", "1"), r.get("dsa_qt", "1"), "w" + r.get("dsa_win", "0"))
    g.setdefault(key, {})[r["seed"]] = r
for key, v in sorted(g.items()):
    v = list(v.values())
    print(f"{' '.join(key):40s} n={len(v)} loss {sum(r['wloss'] for r in v)/len(v):.3f} kept {sum(wt[s]*r['kept'][s] for r in v for s in S.SKILLS)/tot/len(v):5.1f}")
