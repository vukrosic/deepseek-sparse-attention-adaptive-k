# DeepSeek sparse attention where each token decides how many tokens to read

**DeepSeek sparse attention lets each token read only a fixed number of earlier tokens, chosen by a small learned picker. I let each token choose its own number instead, rounded up to 4, 8, 16, 32 or 64 so GPU kernels still see fixed sizes. When tokens are read in GPU-friendly blocks, this works much better than one fixed number: reading 7.6 tokens on average, it gets close to the loss of a fixed 16.**

![Fixed vs chosen number of tokens, for single tokens, blocks and blocks + query tiles](main_result.png)

## The core idea

When a token looks back at earlier tokens, attention reads every one of them. The token doing the looking is called the **query**. For long texts this is the slowest part of a language model.

**DeepSeek sparse attention** ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)) reads only some of them. A small network called the **picker** (the lightning indexer) scores every earlier token, and attention reads only the **top-k**: 2,048 tokens, the same number for every query.

But queries need different amounts. Continuing a common phrase needs the last word or two. Answering a question about something said far back needs a lot of text.

My question: if each query decides how many tokens to read, does the model do better for the same amount of reading?

## What I changed

The picker is trained to copy where the real attention looks. So its scores also show how spread out each query's attention is.

Each query keeps its top-scored tokens until they hold half of the picker's probability (**top-p**). The count is rounded up to 4, 8, 16, 32 or 64, because GPU kernels need a few fixed sizes.

## Results

- **At small budgets it beats a fixed number.** Reading 5.8 tokens on average, it scores 1.339. Fixed 4 tokens score 1.481 and fixed 8 score 1.314.
- **The gain is biggest when tokens are read in blocks**, as GPU kernels do. A fixed 8 tokens in blocks of 4 scores 1.509. A chosen number of blocks scores 1.371 reading 7.6 tokens, close to a fixed 16 (1.352).
- **A minimum of 4 matters; rounding is free.** Allowing 1 or 2 tokens breaks the model (1.566).
- **The rule barely matters.** Top-p, a ratio to the best score and an entropy rule tie.

## Where it stops working

- From about 10 tokens per query on, a chosen and a fixed number give the same loss.
- Letting 4 neighbouring queries share one choice breaks lookup tasks (1.713), unless tokens are read in blocks.
- This is a tiny model with 64-token sequences and 3 seeds per setting (seeds differ by 0.02–0.04). I measured loss, not speed. Whether it holds in large models is a guess.

## Is this new?

[NSA](https://arxiv.org/abs/2502.11089) reads blocks with a fixed count. [Twilight](https://arxiv.org/abs/2502.02770) and [RTPurbo](https://arxiv.org/abs/2605.16928) let each query choose with top-p, without a minimum or rounding. I found no work on rounding to fixed sizes with a minimum, or on the larger gain with blocks.

## Setup

A 2-layer transformer (width 64, mixture of experts, Muon) trained for 18 seconds on a MacBook CPU on 8 small tasks, such as next-word prediction, copying and finding a word's partner. [RESULTS.md](RESULTS.md) has every experiment and number, [toy/](toy/) the code, and [chart.py](chart.py) the charts.
