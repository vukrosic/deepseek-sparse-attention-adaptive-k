# DeepSeek sparse attention where each token decides how many tokens to read

**When each token decides how many earlier tokens to read, DeepSeek sparse attention reads about half as many tokens for almost the same quality.**

**It fits how GPUs work. A GPU loads earlier tokens in blocks of neighbouring tokens, 64 or 128 at a time. So each token picks whole blocks, and the number it reads is always a whole number of blocks, for example 1,024, 1,536 or 2,048 tokens with blocks of 128.**

![A chosen number of blocks reads about half as many tokens as a fixed number for the same loss](blocks_long.png)

## The core idea

When a token looks back at earlier tokens, attention reads every one of them. The token doing the looking is called the **query**. For long texts this is the slowest part of a language model.

**DeepSeek sparse attention** ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)) reads only some of them. A small network called the **picker** (the lightning indexer) scores every earlier token, and attention reads only the **top-k**: 2,048 tokens, the same number for every query.

But queries need different amounts. Continuing a common phrase needs the last word or two. Answering a question about something said far back needs a lot of text.

My question: if each query decides how many tokens to read, does the model do better for the same amount of reading?

## What I changed

The picker is trained to copy where the real attention looks. So its scores also show how spread out each query's attention is.

Each query keeps its top-scored blocks until they hold a share p of the picker's probability (**top-p**). A query whose attention sits in one place reads few blocks; a query whose attention is spread out reads many. In this small model a block is 4 tokens, standing in for the 64- or 128-token blocks of real GPU kernels.

## Results

- **About half the reading for the same loss.** With 128-token sequences (32 blocks), a chosen number of blocks reads 24 tokens on average for a loss of 1.458. A fixed 64 tokens scores 1.449, and normal attention 1.418.
- **Blocks need it most.** With 64-token sequences, a fixed 8 tokens in blocks scores 1.509, against 1.314 for 8 single tokens. A chosen number of blocks brings it back to 1.371.
- **A minimum size matters; powers of 2 do not.** Letting a query read only 1 or 2 tokens breaks the model (1.566). Any whole number of blocks works about as well as powers of 2 (1.403 vs 1.371, close to the noise).
- **The rule barely matters.** Top-p, a ratio to the best score and an entropy rule tie.

## Where it stops working

- With single tokens instead of blocks, on 64-token sequences, the gain disappears from about 10 tokens per query on.
- Letting 4 neighbouring queries share one choice breaks lookup tasks (1.713), unless tokens are read in blocks.
- This is a tiny model with sequences of 64 and 128 tokens, blocks of 4 and 3 seeds per setting; runs of one setting differ by up to 0.1. I measured loss, not speed. Whether it holds in large models with blocks of 64 or 128 is a guess.

## Is this new?

[NSA](https://arxiv.org/abs/2502.11089) reads blocks, with the same count for every query. [Twilight](https://arxiv.org/abs/2502.02770) and [RTPurbo](https://arxiv.org/abs/2605.16928) let each query choose with top-p, over single tokens and without a minimum. I found no work on a per-query count of whole GPU blocks, or on the larger gain it gives with blocks.

## Setup

A 2-layer transformer (width 64, mixture of experts, Muon) trained on a MacBook CPU, 18–49 seconds per run, on 8 small tasks such as next-word prediction, copying and finding a word's partner. [RESULTS.md](RESULTS.md) has every experiment and number, [toy/](toy/) the code, and [chart.py](chart.py) the charts.
