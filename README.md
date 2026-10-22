# DeepSeek sparse attention where each token decides how many tokens to read

**DeepSeek sparse attention lets each token read only a fixed number of earlier tokens, chosen by a small learned picker. I let each token choose its own number instead, rounded up to 4, 8, 16, 32 or 64 so GPU kernels still see fixed sizes. When tokens are read in GPU-friendly blocks, this works much better than one fixed number: reading 7.6 tokens on average, it gets close to the loss of a fixed 16.**

![Fixed vs chosen number of tokens, for single tokens, blocks and blocks + query tiles](main_result.png)

## Contents

- [Background](#background)
  - [How attention reads earlier tokens](#how-attention-reads-earlier-tokens)
  - [How DeepSeek sparse attention picks tokens](#how-deepseek-sparse-attention-picks-tokens)
  - [The question](#the-question)
- [What I changed](#what-i-changed)
  - [Each query chooses how many tokens to read](#each-query-chooses-how-many-tokens-to-read)
  - [Rounding up to a few fixed sizes](#rounding-up-to-a-few-fixed-sizes)
- [Experiments](#experiments)
  - [Fixed number of tokens](#fixed-number-of-tokens)
  - [Training the picker: copying vs task loss](#training-the-picker-copying-vs-task-loss)
  - [Chosen number vs fixed number](#chosen-number-vs-fixed-number)
  - [Which sizes to allow](#which-sizes-to-allow)
  - [Which rule picks the size](#which-rule-picks-the-size)
  - [Picking blocks of tokens](#picking-blocks-of-tokens)
  - [Sharing one choice across neighbouring queries](#sharing-one-choice-across-neighbouring-queries)
  - [Always keeping the last tokens](#always-keeping-the-last-tokens)
- [Where it works](#where-it-works)
  - [At small budgets](#at-small-budgets)
  - [When tokens are picked in blocks](#when-tokens-are-picked-in-blocks)
  - [In a GPU-friendly layout](#in-a-gpu-friendly-layout)
- [Where it does not work](#where-it-does-not-work)
  - [At larger budgets](#at-larger-budgets)
  - [With sizes below 4](#with-sizes-below-4)
  - [When queries share a choice without blocks](#when-queries-share-a-choice-without-blocks)
  - [What I did not test](#what-i-did-not-test)
- [Conclusions](#conclusions)
- [Is this new?](#is-this-new)
  - [What already exists](#what-already-exists)
  - [What I found no published work on](#what-i-found-no-published-work-on)
- [Setup](#setup)
- [Files](#files)

## Background

### How attention reads earlier tokens

When a token looks back at earlier tokens, attention reads every one of them and takes a weighted mix. The token doing the looking is called the **query**.

For long texts this reading is the slowest part of a language model, because every query reads every earlier token.

### How DeepSeek sparse attention picks tokens

**DeepSeek sparse attention** ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)) reads only some of the earlier tokens. A small, fast network called the **picker** (DeepSeek calls it the lightning indexer) gives every earlier token a score.

The picker is a tiny attention: 4 heads of size 16, score = sum over heads of weight × ReLU(query · key). It costs much less than the real attention.

Attention then reads only the **top-k**: the k tokens with the highest scores. DeepSeek uses k = 2,048, the same number for every query.

The picker cannot learn from the model's loss directly, because "keep or drop" has no gradient. DeepSeek trains it by **copying**: the picker's scores, turned into probabilities, are pushed toward where the real attention looks (a KL loss). For the first steps attention stays dense, so the picker learns before it is used.

### The question

Queries need different amounts. A token that continues a common phrase needs the last word or two. A token that answers a question about something said far back must search a lot of text.

So: if each query decides how many tokens to read, does the model do better for the same amount of reading, and can it still run fast on a GPU?

## What I changed

### Each query chooses how many tokens to read

Because the picker copies the attention, its probabilities estimate where the attention wants to look. So the picker can also tell how spread out each query's attention is.

The rule: each query keeps its top-scored tokens until they hold half of the picker's probability. A query whose attention sits on one token keeps few. A query whose attention is spread out keeps many. This is **top-p** selection, the same idea as top-p sampling of words.

### Rounding up to a few fixed sizes

GPU kernels are written for fixed sizes. A few fixed sizes are much easier to make fast than any count from 1 to 64.

So the chosen count is **rounded up** to 4, 8, 16, 32 or 64. A query that wants 5 tokens reads 8.

## Experiments

All numbers are validation loss (lower is better) and the average number of tokens each query reads. Every setting has 3 seeds; seeds differ by 0.02–0.04, so differences under about 0.03 are noise. Per-seed numbers are in [RESULTS.md](RESULTS.md).

### Fixed number of tokens

First, DeepSeek's fixed top-k, as the baseline. Normal attention, reading everything, scores 1.257.

- top-16: 1.277
- top-8: 1.314
- top-4: 1.481

So 16 of 64 tokens is almost free, and 4 is clearly worse. The room for a chosen size is at small budgets.

### Training the picker: copying vs task loss

I tested 7 ways to train the picker, from pure copying to pure task loss. For task loss, the picker's scores are added to the attention scores so the model's loss reaches them.

![Copying the attention beats training on the task](picker_training.png)

Copying wins. Every amount of task loss makes it worse. With task loss only, all the picker's ReLU scores in the first layer fell to zero, so it picked tokens at random.

This is why the chosen size works: a picker trained by copying knows where the attention looks.

### Chosen number vs fixed number

Reading 5.8 tokens on average, the chosen size scores 1.339. Fixed top-4 scores 1.481 and fixed top-8 (7.7 tokens on average, since early tokens have fewer than 8 to read) scores 1.314. At the same average, the chosen size is better.

Tasks choose very different amounts. Finding a word's partner among 30 pairs reads 17 tokens. Predicting the next word of text reads 5.

### Which sizes to allow

![The smallest size matters, rounding costs nothing](sizes.png)

The smallest allowed size matters most. If a query may read only 1 or 2 tokens, the model breaks (1.566). The picker is sometimes very sure about the wrong token, and a minimum of 4 is a safety margin.

Rounding up to 4, 8, 16, 32 or 64 costs nothing: it scores the same as allowing any count from 4 (1.339 vs 1.357).

### Which rule picks the size

I tried two more rules besides top-p:

- **ratio:** keep every token that scores at least r times the best one;
- **entropy:** keep c times the picker's effective number of tokens.

![Three rules give about the same result](rules.png)

All three land on about the same curve. So the rule does not matter much; reading each query's own needs does.

### Picking blocks of tokens

GPU kernels for sparse attention do not load single tokens. They load **blocks** of neighbouring tokens.

So the picker scores blocks of 4 neighbouring tokens (the sum of its probabilities inside the block) and keeps whole blocks, as in [NSA](https://arxiv.org/abs/2502.11089).

- Blocks make a fixed number much worse: 2 blocks (8 tokens) score 1.509, against 1.314 for 8 single tokens.
- A chosen size repairs most of it: chosen blocks score 1.371 while reading 7.6 tokens, close to a fixed 4 blocks (16 tokens, 1.352).

### Sharing one choice across neighbouring queries

GPU kernels also process several neighbouring queries together. If they want the same blocks, the kernel loads those blocks once.

So 4 neighbouring queries, a **query tile**, share one choice and one size, from the average of their picker probabilities.

- With blocks it is nearly free: 1.371 → 1.378, reading 9.1 tokens. A fixed size with the same layout scores 1.569 at 8 tokens and 1.325 at 16.
- Without blocks it breaks the model: 1.713. Copying (loss 0.60 → 1.54) and finding a word's partner (0.77 → 2.43) fail. A query that needs one far token shares its choice with 3 neighbours that want other tokens.

### Always keeping the last tokens

NSA always reads the last few tokens, its sliding window. I tested always keeping the last 4.

- With chosen single tokens: 1.315 while reading 10.0, no better than reading more chosen tokens.
- With chosen blocks: 1.387 while reading 12.1, no better than blocks alone.
- With query tiles and no blocks: 1.713 → 1.561, so it repairs only part of the damage.

## Where it works

### At small budgets

Below about 8 tokens per query, a chosen size beats a fixed one at the same average, with every rule I tried.

### When tokens are picked in blocks

This is where the chosen size helps most. A fixed number of blocks loses a lot against single tokens (1.509 vs 1.314 at 8). A chosen number of blocks gets back most of that: 1.371 at 7.6 tokens.

### In a GPU-friendly layout

Blocks of tokens, tiles of queries and a chosen size rounded up to fixed values together score 1.378 at 9.1 tokens. The same layout with a fixed 8 tokens scores 1.569.

## Where it does not work

### At larger budgets

From about 10 tokens per query on, a chosen size and a fixed size give the same loss.

### With sizes below 4

Letting a query read only 1 or 2 tokens breaks the model (1.566), whatever the rule.

### When queries share a choice without blocks

Query tiles over single tokens break copying and lookup (1.713). An always-kept window repairs only part of it. Why blocks repair it, I did not test.

### What I did not test

- **Speed.** I measured loss, not time. That fixed sizes, blocks and tiles are faster comes from how existing kernels work ([NSA](https://arxiv.org/abs/2502.11089), [FSA](https://arxiv.org/abs/2508.18224)).
- **Real model sizes.** This is a 2-layer model with 64-token sequences. A real model reads hundreds or thousands of tokens per query, in blocks of 64. Whether the same pattern holds there is a guess.
- **Block cost.** Picking in blocks still loses against single tokens at the same count (1.371 vs about 1.32), and I did not try to close that gap.

## Conclusions

- A picker trained by copying is a good estimate of how spread out each query's attention is, so it can choose the size too, at no extra cost.
- The rule that turns this into a size does not matter much: top-p, ratio and entropy tie.
- Keep a minimum size of 4. Rounding up to 4, 8, 16, 32 or 64 is free.
- The gain is largest when tokens are picked in blocks, which is how GPU kernels already work.
- The layout I would keep: **blocks of tokens, tiles of queries, and a per-query size rounded up to a few fixed values with a minimum.**

## Is this new?

### What already exists

- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556): the picker, trained by copying, with a fixed top-k of single tokens.
- [NSA](https://arxiv.org/abs/2502.11089) and [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392): picking blocks of tokens, with a fixed count; MiniMax also always keeps a local block.
- [Twilight](https://arxiv.org/abs/2502.02770): top-p budgets on top of an existing sparse method, without training.
- [Full Attention Strikes Back (RTPurbo)](https://arxiv.org/abs/2605.16928): a small picker trained by copying, with top-p per query (p = 0.9), when converting a dense model. It describes no minimum size and no rounding.

### What I found no published work on

- Rounding each query's budget up to a few fixed sizes, and the finding that the minimum size matters more than the rounding.
- The finding that a per-query budget gains much more when tokens are picked in blocks.
- The finding that sharing one choice across neighbouring queries breaks lookup tasks unless picking is in blocks.

## Setup

A small transformer (2 layers, width 64, mixture of experts, Muon optimizer) trained for about 18 seconds on a MacBook CPU. Like a language model, it learns 8 small tasks mixed into one stream: predicting the next word of text, copying, common and rare facts, repeating a token from 2 back, finding a word's partner among 30 pairs, sorting and counting. Sequences are 64 tokens long, and every setting is run with 3 seeds.

## Files

- [RESULTS.md](RESULTS.md) has every number per seed, the knobs and the command for each run.
- [toy/](toy/) has the model, tasks and training code; `toy/run.sh SEED` runs one setting.
- [chart.py](chart.py) draws the four charts.
