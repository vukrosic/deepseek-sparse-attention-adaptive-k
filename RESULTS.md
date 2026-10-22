# Results

**Model:** 2-layer transformer, width 64, 2 attention heads, pre-norm RMSNorm, RoPE, QK-norm, gated attention, SwiGLU experts, a value embedding, an 8-expert mixture of experts (2 active), short convolution on the keys. Muon on all matrices, AdamW on embeddings and norms. Sequences of 64 tokens, batch 32, 260 steps, lr 0.028. About 18 s per run on a MacBook CPU.

**Tasks** (one per sequence, mixed like a language model's data): next word in text from a fixed random chain, copy a piece, common fact, rare fact, repeat the token 2 back, find a word's partner among 30 pairs, sort, count zeros.

**Loss:** test loss weighted by how often each task appears in training (lower is better). Two runs that differ only in seed differ by about 0.02–0.04.

**Normal attention:** 1.221, 1.267, 1.284 on seeds 0–2 (mean 1.257); over 9 seeds the mean is 1.268.

**Tokens read per word:** at test time, the average number of tokens each query attends to, over both layers. Each task is weighted by its share of scored positions (text and repeat-2-back score 62–63 positions per sequence, copy 16.5, sort 6, the others 1). Fact questions are 3 tokens long, so they can read at most 3.

## DeepSeek sparse attention as built here

- **Picker** (DeepSeek calls it the lightning indexer): 4 heads of size 16 with RoPE. Score of earlier token s for query t: I(t, s) = Σ_heads w · ReLU(q · k). Its inputs are detached, so it never changes the rest of the model.
- **Selection:** each query keeps its top-k earlier tokens by I and attends only to them, in both layers.
- **Training the picker:** KL(main attention summed over heads ‖ softmax(I)) over the kept tokens. The first 60 steps use dense attention while the picker learns. The picker's matrices are trained with Muon.
- **Chosen budget:** take softmax(I) per query, then pick a size with one of the rules below, and round it up to the next allowed size (default 4, 8, 16, 32, 64).

## Knobs

All in `experiments/suite.py`. `experiments/run.sh` sets the model above, sequences of 64 and DeepSeek sparse attention with a 60-step warm-up and a Muon-trained picker.

| knob | meaning |
|---|---|
| `DSA_K=8` | fixed top-k: 8 tokens per query, used when `DSA_P` is unset (`run.sh` defaults to 64) |
| `DSA_P=0.5` | chosen budget: the rule's parameter (p, r or c) |
| `DSA_RULE=p` | `p`: smallest count whose top tokens hold p of softmax(I). `ratio`: tokens with softmax(I) ≥ r × the best. `ent`: c × exp(entropy of softmax(I)) |
| `DSA_BUCKETS=4,8,16,32,64` | allowed sizes; the count is rounded up to the next one. A single value (e.g. `8`) is a fixed size through the same code |
| `DSA_BLK=4` | the picker scores blocks of 4 neighbouring tokens (softmax(I) summed inside the block) and keeps whole blocks |
| `DSA_QT=4` | 4 neighbouring queries share one choice and one size (their softmax(I) averaged) |
| `DSA_WIN=4` | the last 4 tokens are always kept as well |
| `DSA_TRAIN=copy` | how the picker learns: `copy`, `copyall`, `sum`, `alt`, `anneal`, `switch`, `task` (see Step 2) |

Command for every run: `cd experiments && env [knobs] ./run.sh SEED`. Then `python3 summ.py` prints mean loss and tokens read per setting.

## Step 1: fixed top-k

`env DSA_K=k ./run.sh SEED` (with `DSA_P` unset, `DSA_K` is the fixed count)

| k | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|
| 4 | 1.478 | 1.452 | 1.514 | 1.481 | 4.0 |
| 8 | 1.331 | 1.292 | 1.318 | 1.314 | 7.7 |
| 16 | 1.278 | 1.263 | 1.288 | 1.276 | 14.3 |
| 32 | 1.265 | 1.253 | 1.278 | 1.265 | 24.7 |

## Step 2: how the picker is trained (fixed top-8)

`env DSA_K=8 DSA_TRAIN=mode ./run.sh SEED`

| mode | what it does | s0 | s1 | s2 | mean |
|---|---|---|---|---|---|
| `copy` | KL to the main attention over kept tokens (DeepSeek) | 1.331 | 1.292 | 1.318 | 1.314 |
| `copyall` | KL over all earlier tokens | 1.302 | 1.283 | 1.318 | 1.301 |
| `sum` | copy, plus log softmax(I) added to the attention scores so the task loss also trains it | 1.365 | 1.398 | 1.420 | 1.394 |
| `alt` | task loss always, copy on every other step | 1.348 | 1.485 | 1.460 | 1.431 |
| `anneal` | task loss always, copy weight falls from 1 to 0 | 1.451 | 1.463 | 1.473 | 1.462 |
| `switch` | copy during warm-up, then task loss only | 2.088 | 2.380 | 2.407 | 2.292 |
| `task` | task loss only | 2.525 | 2.513 | 2.558 | 2.532 |

With `task`, every picker score in layer 1 ends at 0 (the ReLU outputs are all zero), so top-k picks arbitrary tokens.

## Step 3: chosen budget, top-p rounded up to 4/8/16/32/64

`env DSA_P=p ./run.sh SEED`

| p | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|
| 0.5 | 1.331 | 1.317 | 1.368 | 1.339 | 5.8 |
| 0.7 | 1.286 | 1.312 | 1.331 | 1.310 | 9.8 |
| 0.8 | 1.262 | 1.255 | 1.307 | 1.275 | 14.0 |
| 0.9 | 1.257 | 1.267 | 1.289 | 1.271 | 18.5 |

Tokens read per task at p=0.5 (mean of 3 seeds): common fact 3.0, rare fact 3.0, sort 4.2, text 4.9, copy 6.2, count 8.7, repeat 2 back 9.2, find a partner 17.3.

## Step 4: which sizes are allowed (p=0.5)

`env DSA_P=0.5 DSA_BUCKETS=list ./run.sh SEED`

| allowed sizes | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|
| 4, 8, 16, 32, 64 | 1.331 | 1.317 | 1.368 | 1.339 | 5.8 |
| any count from 4 (4, 5, …, 64) | 1.355 | 1.336 | 1.379 | 1.357 | 5.3 |
| 4, 16, 64 | 1.300 | 1.326 | 1.344 | 1.324 | 7.0 |
| 8, 16, 32, 64 | 1.282 | 1.274 | 1.301 | 1.286 | 9.5 |
| 1, 2, 4, 8, 16, 32, 64 | 1.543 | 1.588 | 1.568 | 1.566 | 3.5 |
| any count from 1 (1, 2, …, 64) | 1.527 | 1.585 | 1.638 | 1.583 | 3.0 |

Any count from 1 at p=0.7: 1.385, 1.400, 1.403 (mean 1.396, 5.1 tokens).

## Step 5: how each query's size is chosen (sizes 4..64)

`env DSA_RULE=rule DSA_P=value ./run.sh SEED`

| rule | value | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|---|
| ratio | 0.3 | 1.322 | 1.288 | 1.365 | 1.325 | 6.4 |
| ratio | 0.2 | 1.285 | 1.284 | 1.335 | 1.301 | 7.1 |
| ratio | 0.1 | 1.281 | 1.257 | 1.300 | 1.279 | 8.5 |
| entropy | 0.25 | 1.344 | 1.357 | 1.368 | 1.356 | 5.2 |
| entropy | 0.5 | 1.288 | 1.286 | 1.328 | 1.301 | 7.6 |
| entropy | 1 | 1.255 | 1.253 | 1.296 | 1.268 | 13.9 |
| entropy | 2 | 1.252 | 1.252 | 1.299 | 1.268 | 19.4 |

Top-p is in Step 3. Below 8 tokens read, every rule beats the fixed top-k line. From 10 tokens on, all rules and fixed top-k agree within noise.

## Step 6: GPU-friendly picking (top-p, p=0.5)

`env DSA_P=0.5 DSA_BLK=b DSA_QT=q DSA_WIN=w [DSA_BUCKETS=8 or 16 for a fixed size] ./run.sh SEED`

| picking | size | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|---|
| single tokens | chosen | 1.331 | 1.317 | 1.368 | 1.339 | 5.8 |
| single tokens | fixed 8 | 1.331 | 1.292 | 1.318 | 1.314 | 7.7 |
| blocks of 4 | chosen | 1.381 | 1.344 | 1.390 | 1.371 | 7.6 |
| blocks of 4 | fixed 8 | 1.471 | 1.510 | 1.545 | 1.509 | 7.7 |
| blocks of 4 | fixed 16 | 1.344 | 1.313 | 1.398 | 1.352 | 14.3 |
| blocks of 4 + tiles of 4 | chosen | 1.349 | 1.392 | 1.394 | 1.378 | 9.1 |
| blocks of 4 + tiles of 4 | fixed 8 | 1.578 | 1.544 | 1.586 | 1.569 | 7.7 |
| blocks of 4 + tiles of 4 | fixed 16 | 1.301 | 1.321 | 1.352 | 1.325 | 14.3 |
| tiles of 4, single tokens | chosen | 1.696 | 1.687 | 1.754 | 1.713 | 8.4 |
| tiles of 4 + window 4 | chosen | 1.596 | 1.546 | 1.540 | 1.561 | 12.0 |
| single tokens + window 4 | chosen | 1.321 | 1.301 | 1.323 | 1.315 | 10.0 |
| blocks of 4 + window 4 | chosen | 1.374 | 1.370 | 1.416 | 1.387 | 12.1 |

Per-task loss, mean of 3 seeds (chosen size):

| picking | text | copy | common fact | repeat 2 back | find a partner | sort | rare fact | count |
|---|---|---|---|---|---|---|---|---|
| single tokens | 1.98 | 0.60 | 0.26 | 0.26 | 0.77 | 3.87 | 0.64 | 2.51 |
| blocks of 4 | 1.98 | 0.75 | 0.20 | 0.27 | 0.92 | 3.89 | 0.60 | 2.51 |
| tiles of 4 | 1.98 | 1.54 | 0.30 | 0.73 | 2.43 | 4.14 | 0.85 | 2.61 |
| blocks of 4 + tiles of 4 | 1.98 | 0.93 | 0.19 | 0.28 | 0.55 | 3.92 | 0.64 | 2.55 |
| tiles of 4 + window 4 | 1.98 | 1.33 | 0.23 | 0.37 | 1.60 | 4.10 | 0.78 | 2.51 |

Tiles without blocks break copying and partner lookup. Why blocks repair this is not tested.

Sizes as any whole number of blocks instead of powers of 2 (`DSA_BUCKETS=4,8,12,...,64`, blocks of 4, chosen size): 1.400, 1.396, 1.412, mean 1.403, reading 7.5 tokens. Powers of 2 score 1.371 at 7.6, so finer sizes give no gain here.

## Step 7: longer sequences, many blocks per token (128 tokens, blocks of 4)

In Steps 1–6 a token reads only about 2 blocks. DeepSeek reads 2,048 tokens, which is 32 blocks of 64. Here sequences are 128 tokens (32 blocks of 4), so a token reads 4 to 16 blocks. The long tasks (finding a partner, counting) fill the longer sequence, so losses are not comparable with Steps 1–6. Each run takes about 49 s.

`env SEQ=128 DSA_K=128 DSA_BLK=4 DSA_P=p DSA_BUCKETS=... ./run.sh SEED`: fixed size = one value in `DSA_BUCKETS`; chosen size = `DSA_BUCKETS=4,8,12,...,128` (any whole number of blocks). Normal attention = `env SEQ=128 DSA_K=128 ./run.sh SEED`.

| size | s0 | s1 | s2 | mean | tokens read |
|---|---|---|---|---|---|
| normal attention | 1.435 | 1.415 | 1.402 | 1.418 | 64.6 |
| fixed 16 (4 blocks) | 1.673 | 1.570 | 1.687 | 1.643 | 15.1 |
| fixed 32 (8 blocks) | 1.564 | 1.516 | 1.525 | 1.535 | 28.4 |
| fixed 64 (16 blocks) | 1.482 | 1.438 | 1.426 | 1.449 | 48.8 |
| chosen, p = 0.5 | 1.597 | 1.560 | 1.562 | 1.573 | 13.6 |
| chosen, p = 0.7 | 1.496 | 1.452 | 1.427 | 1.458 | 24.2 |
| chosen, p = 0.85 | 1.472 | 1.419 | 1.437 | 1.443 | 32.1 |

A chosen number of blocks reading 24–32 tokens matches a fixed 64 tokens (1.443–1.458 vs 1.449). Unlike Steps 3–5, the gain does not vanish at larger budgets here: it holds up to at least 16 blocks. Runs of one setting differ by up to 0.12 (fixed 16).

## Step 7b: the same at 256 and 512 tokens (blocks of 4)

`cd experiments && ./blocks_sweep.sh SEQ SEED "chosen p values" "fixed sizes"` ("full" = normal attention); `python3 blocks_summ.py SEQ` prints the table. Chosen sizes = any whole number of blocks up to SEQ. About 2 min per run at 256 tokens, 6–7 min at 512.

256 tokens (64 blocks of 4):

| size | s0 | s1 | mean | tokens read |
|---|---|---|---|---|
| normal attention | 1.713 | 1.640 | 1.676 | 128.7 |
| fixed 32 (8 blocks) | 1.855 | 1.815 | 1.835 | 30.2 |
| fixed 64 (16 blocks) | 1.776 | 1.705 | 1.741 | 56.6 |
| fixed 128 (32 blocks) | 1.744 | 1.632 | 1.688 | 97.4 |
| chosen, p = 0.5 | 1.785 | 1.835 | 1.810 | 29.1 |
| chosen, p = 0.7 | 1.727 | 1.626 | 1.676 | 50.2 |
| chosen, p = 0.85 | 1.676 | 1.617 | 1.646 | 69.2 |

Chosen p = 0.7 reads 50 tokens and ties normal attention (1.676), slightly better than fixed 128 reading 97 (1.688). Chosen p = 0.85 reading 69 is the best row (1.646). At the smallest size (about 30 tokens) choosing ties the fixed amount (1.810 vs 1.835, within noise). Runs of one setting differ by up to 0.11.

512 tokens, seed 0 only: normal attention 2.962 (258 read), fixed 256 2.931 (195), fixed 128 2.943 (113), chosen p = 0.7 2.990 (120). Reading everything is no better than reading 113 tokens, so the model has not learned to use the long text in 260 steps, and this length cannot separate the versions.

## Step 8: the best minimum vs the context length (single tokens, p=0.5)

Same model and training (260 steps). Context lengths 32, 64, 128, 256, 512. Above 16 tokens, the copy, repeat-2-back, find-a-partner and count tasks fill the whole text, so they get harder as texts get longer; the fact and sort tasks stay the same short texts.

Chosen size: allowed sizes = the minimum, then powers of 2 up to the context length. Fixed size: every token reads the same k tokens (DeepSeek).

`cd experiments && ./context_sweep.sh LENGTH SEED "minimums" "fixed sizes"`, then `python3 context_summ.py LENGTH`.

"vs fixed" = loss minus the loss of DeepSeek's fixed amount reading the same number of tokens on average (straight line between the two nearest fixed sizes on a log scale of tokens read). Negative = better than fixed.

| length | setting | s0 | s1 | mean | tokens read | vs fixed |
|---|---|---|---|---|---|---|
| 32 | fixed 4 / 8 / 16 | 1.441 / 1.343 / 1.296 | 1.492 / 1.347 / 1.324 | 1.467 / 1.345 / 1.310 | 3.9 / 7.4 / 12.7 | |
| 32 | min 1 | 1.881 | 1.794 | 1.837 | 2.3 | +0.269 |
| 32 | min 2 | 1.502 | 1.463 | 1.482 | 3.1 | −0.029 |
| 32 | min 4 | 1.399 | 1.393 | 1.396 | 4.9 | −0.028 |
| 32 | min 8 | 1.328 | 1.342 | 1.335 | 7.9 | −0.006 |
| 64 | fixed 4 / 8 / 16 | 1.478 / 1.331 / 1.278 | 1.452 / 1.292 / 1.263 | 1.465 / 1.312 / 1.271 | 4.0 / 7.7 / 14.3 | |
| 64 | min 1 | 1.543 | 1.588 | 1.565 | 3.8 | +0.088 |
| 64 | min 2 | 1.387 | 1.375 | 1.381 | 4.1 | −0.078 |
| 64 | min 4 | 1.331 | 1.317 | 1.324 | 5.9 | −0.050 |
| 64 | min 8 | 1.282 | 1.274 | 1.278 | 9.7 | −0.019 |
| 128 | fixed 4 / 8 / 16 | 1.953 / 1.714 / 1.585 | 1.956 / 1.653 / 1.578 | 1.955 / 1.684 / 1.582 | 4.0 / 7.8 / 15.2 | |
| 128 | min 1 | 1.808 | 1.719 | 1.764 | 5.9 | −0.033 |
| 128 | min 2 | 1.729 | 1.590 | 1.659 | 7.1 | −0.063 |
| 128 | min 4 | 1.628 | 1.544 | 1.586 | 8.3 | −0.089 |
| 128 | min 8 | 1.571 | 1.492 | 1.532 | 12.8 | −0.076 |
| 256 | fixed 16 / 32 | 2.236 / 1.844 | 1.848 / 1.699 | 2.042 / 1.772 | 15.6 / 30.3 | |
| 256 | min 1 | 2.134 | 1.929 | 2.032 | 16.3 | +0.008 |
| 256 | min 2 | 1.882 | 1.914 | 1.898 | 17.8 | −0.090 |
| 256 | min 4 | 1.890 | 1.745 | 1.817 | 21.4 | −0.096 |
| 256 | min 8 | 1.775 | 1.728 | 1.751 | 22.0 | −0.151 |
| 256 | min 16 | 1.779 | 1.737 | 1.758 | 29.3 | −0.028 |
| 512 | fixed 32 / 64 | 3.051 / 2.816 | – | | 31.1 / 60.4 | |
| 512 | min 8 | 2.871 | – | | 52.4 | +0.004 (s0) |
| 512 | min 16 | 2.787 | 2.358 | 2.573 | 56.2 | −0.058 (s0) |
| 512 | min 32 | 2.474 | 2.238 | 2.356 | 66.9 | −0.294 (s0) |
| 512 | min 64 | 2.683 | – | | 88.5 | +0.002 (s0) |

At 512 only seed 0 has fixed runs (8 minutes per run), so "vs fixed" uses seed 0. Seed 1 confirms min 32 beats min 16 (2.238 vs 2.358).

Best minimum: 2 (32 tokens; 2 and 4 tie), 2 (64), 4 (128), 8 (256), 32 (512). A straight line in log-log fitted to these points has slope 1.0: best minimum ≈ length / 24.

Where the gain comes from (per-task loss, 512 tokens, min 8 → 16 → 32): copy 4.51 → 3.35 → 2.64, repeat 2 back 1.89 → 1.36 → 1.03, find a partner 5.16 → 4.60 → 3.97. Count (5.2–5.4) and the short tasks do not change. These are look-up tasks; at longer lengths the picker misses the right token more often, and a larger minimum still includes it.

Predictions made before running, and what happened:
- From 32–128, the copy-task gap between min 1 and min 4 shrank (0.64, 0.35, 0.20); predicted 0.11 at 256. Measured 0.23: **wrong**, the gap stops shrinking.
- From best minimums 2, 2, 4, 8 (rule length / 32): predicted min 16 is no better than min 8 at 256. Measured 1.758 vs 1.751: **right**.
- Same rule predicted a best minimum of 16 at 512. Measured 32: **wrong by a factor of 2**, which moved the fitted rule to length / 24.

Caveats: at longer lengths the model learns badly in 260 steps (loss rises from 1.3 to 2.4), and runs of one setting differ by up to 0.4 at 256–512 tokens.

## Step 9: calculated attention compute in DeepSeek-V3.2 (not measured)

Sizes from DeepSeek-V3.2-Exp `config.json`: 128 attention heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64, picker (`index_n_heads`) 64 heads of width 128, top-k 2,048.

Per new token, per layer, per earlier token:

- reading it (MLA in absorbed form: 576-wide key score + 512-wide value): 2 × 128 × (576 + 512) = 278,528 operations;
- scoring it with the picker: 2 × 64 × 128 = 16,384 operations (17× less).

Cost = 16,384 × L + 278,528 × k, where L is the text length and k the tokens read.

| L | normal attention (k = L) | DeepSeek (k = 2,048) | each token chooses (k = 1,024) | saved |
|---|---|---|---|---|
| 4,096 | 1.14 | 0.638 | 0.352 | 45% |
| 8,192 | 2.28 | 0.705 | 0.419 | 40% |
| 16,384 | 4.56 | 0.839 | 0.554 | 34% |
| 32,768 | 9.13 | 1.107 | 0.822 | 26% |
| 65,536 | 18.25 | 1.644 | 1.359 | 17% |
| 131,072 | 36.51 | 2.718 | 2.433 | 10% |

Billion operations. k = 1,024 assumes the halving from step 7 carries over. Operations, not time: the picker runs in FP8. Chart: `compute_saving.png` (chart.py, chart 8).

## Literature

- DeepSeek-V3.2 ([arXiv 2512.02556](https://arxiv.org/abs/2512.02556)): DeepSeek sparse attention. A lightning indexer (ReLU, trained by KL to the dense attention, dense warm-up first) picks a fixed top-k of single tokens per query (2,048 in the released model).
- NSA ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089)): picks blocks of tokens with a fixed count, plus a compressed branch and a sliding-window branch, with a hardware-aligned kernel.
- FSA ([arXiv 2508.18224](https://arxiv.org/abs/2508.18224)): an NSA kernel that loops over key blocks outside and query tokens inside.
- MiniMax Sparse Attention ([arXiv 2606.13392](https://arxiv.org/abs/2606.13392)): an index branch scores key-value blocks and keeps a fixed top-k per GQA group, trained with a KL loss, with a forced local block.
- Twilight ([arXiv 2502.02770](https://arxiv.org/abs/2502.02770), NeurIPS 2025): training-free top-p pruning on top of an existing sparse method, so the budget varies per query.
- Full Attention Strikes Back / RTPurbo ([arXiv 2605.16928](https://arxiv.org/abs/2605.16928)): converts a dense model in a few hundred steps; a 16-dimensional indexer trained by KL, top-p per head and query (p = 0.9), a block mask of 64 in the kernel. It reports budgets varying about 5× across queries. No minimum size or rounding is described.
- AB-Sparse ([arXiv 2605.12110](https://arxiv.org/abs/2605.12110)): a different block size per head under a fixed token budget.
- The Sparse Frontier ([arXiv 2504.17768](https://arxiv.org/abs/2504.17768)): compares fixed-budget and coverage-threshold sparse attention.
