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

**Loss** measures how wrong the model's guesses of the next token are. Lower is better. Normal attention, which reads every earlier token, scores 1.418 in this test.

### Half the reading for the same quality

I compared two versions of DeepSeek sparse attention on texts of 128 tokens, cut into 32 blocks of 4 tokens.

In the first, every token reads the same number of blocks, as in DeepSeek. Reading 16 blocks (64 tokens), the loss is 1.449.

In the second, each token chooses how many blocks to read. On average a token reads 6 blocks (24 tokens), and the loss is 1.458.

So the second reads less than half as many tokens, and its loss is only 0.009 higher. That gap is smaller than the difference between two runs of the same setting.

### Reading in blocks is where the choice helps most

Reading whole blocks is fast on a GPU, but it wastes reading: if a token needs one word, it must also read the 3 words next to it.

With the same number for every token, this costs a lot. On 64-token texts, reading 8 single tokens gives a loss of 1.314. Reading 8 tokens as 2 blocks gives 1.509, much worse.

When each token chooses its number of blocks, the loss comes back down to 1.371, while reading about the same amount (7.6 tokens).

### Every token must read at least a few tokens

If a token may read only 1 or 2 tokens, the model breaks (loss 1.566). The picker is sometimes sure about the wrong token, and then the token reads nothing useful. A minimum of 4 tokens fixes this.

The sizes do not need to be powers of 2 (4, 8, 16, 32 and so on). Any whole number of blocks works about as well.

### How a token chooses barely matters

I tried three ways to choose: keep the top blocks until they hold a share of the picker's score, keep every block that scores close to the best one, or count how spread out the scores are. All three gave about the same results.

## Where it stops working

When a token reads single tokens instead of blocks, choosing helps only when tokens read very little (under about 10 of 64 tokens). Above that, the same number for every token does just as well.

GPUs also like to process 4 neighbouring tokens together, so I let 4 neighbours share one choice of what to read. Without blocks this breaks the model (loss 1.713): one token needs a word far back, its 3 neighbours want other words, and the shared choice serves none of them well. With blocks, sharing costs almost nothing.

This is a tiny model with texts of 64 and 128 tokens and blocks of 4 tokens. Each setting ran 3 times, and runs of one setting differ by up to 0.1. I measured quality, not speed. Whether this holds in large models with blocks of 64 or 128 tokens is a guess.

## Is this new?

[NSA](https://arxiv.org/abs/2502.11089) reads blocks, with the same count for every query. [Twilight](https://arxiv.org/abs/2502.02770) and [RTPurbo](https://arxiv.org/abs/2605.16928) let each query choose with top-p, over single tokens and without a minimum. I found no work on a per-query count of whole GPU blocks, or on the larger gain it gives with blocks.

## Setup

A 2-layer transformer (width 64, mixture of experts, Muon) trained on a MacBook CPU, 18–49 seconds per run, on 8 small tasks such as next-word prediction, copying and finding a word's partner. [RESULTS.md](RESULTS.md) has every experiment and number, [toy/](toy/) the code, and [chart.py](chart.py) the charts.
