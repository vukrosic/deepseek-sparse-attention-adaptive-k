"""Draws every chart in this repo from the numbers in RESULTS.md (3-seed means). usage: python3 chart.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, PURPLE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#3a9a5b", "#8a5cc2", "#9a9890", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 11,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False})
FULL = 1.257   # normal attention, same 3 seeds

# (tokens kept per word on average, loss)
FIXED = [(4.0, 1.481), (7.7, 1.314), (14.3, 1.277), (24.7, 1.265)]          # top-4, 8, 16, 32
TOP_P = [(5.8, 1.339), (9.8, 1.310), (14.0, 1.275), (18.5, 1.271)]          # p = 0.5, 0.7, 0.8, 0.9
RATIO = [(6.4, 1.325), (7.1, 1.301), (8.5, 1.279)]                         # r = 0.3, 0.2, 0.1
ENT = [(5.2, 1.356), (7.6, 1.301), (13.9, 1.268), (19.4, 1.268)]            # c = 0.25, 0.5, 1, 2
BLK_FIXED, BLK_CHOSEN = [(7.7, 1.509), (14.3, 1.352)], (7.6, 1.371)         # blocks of 4 tokens
BT_FIXED, BT_CHOSEN = [(7.7, 1.569), (14.3, 1.325)], (9.1, 1.378)           # blocks of 4 + tiles of 4 words


def frame(fig, title, sub, note):
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.07, ha="left")
    fig.text(0.07, 0.885, sub, fontsize=11, color=INK2)
    fig.text(0.07, 0.03, note, fontsize=10, color=INK2, ha="left", va="bottom", linespacing=1.5)


def full_line(ax, x):
    ax.axhline(FULL, color=GRAY, lw=1.5, ls="--")
    ax.text(x, FULL - 0.008, f"normal attention {FULL}", ha="right", va="top", fontsize=10, color=INK2)


# 1. main result: fixed vs chosen size, for single tokens, blocks, blocks + query tiles
fig, ax = plt.subplots(figsize=(11.5, 6.6))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22)
frame(fig, "A chosen number of tokens per word helps most when tokens are read in blocks",
      "DeepSeek sparse attention, sequences of 64 tokens. Dashed = same number for every word, star = each word chooses. Mean of 3 runs.",
      "Each word keeps the top tokens that hold half of its picker's attention, rounded up to 4, 8, 16, 32 or 64 tokens.\n"
      "Blocks: the picker keeps whole blocks of 4 neighbouring tokens. Tiles: 4 neighbouring words share one choice. Both match how GPU kernels load data.")
for fixed, chosen, c, name in ((FIXED, TOP_P[0], BLUE, "single tokens"), (BLK_FIXED, BLK_CHOSEN, ORANGE, "blocks of 4 tokens"),
                               (BT_FIXED, BT_CHOSEN, GREEN, "blocks of 4 tokens + tiles of 4 words")):
    ax.plot([a for a, _ in fixed], [b for _, b in fixed], color=c, ls="--", marker="o", ms=6, lw=1.8, label=f"{name}: same number")
    ax.plot(*chosen, color=c, marker="*", ms=19, ls="none", markeredgecolor="white", markeredgewidth=1, label=f"{name}: each word chooses")
    ax.annotate(f"{chosen[1]:.3f}", chosen, textcoords="offset points", xytext=(10, -4), fontsize=10, color=c, fontweight="bold")
for (a, b), lab in zip(BLK_FIXED, ("8", "16")):
    ax.annotate(f"fixed {lab}: {b:.3f}", (a, b), textcoords="offset points", xytext=(-92, -4) if lab == "8" else (8, 4), fontsize=9.5, color=ORANGE)
for (a, b), lab in zip(BT_FIXED, ("8", "16")):
    ax.annotate(f"fixed {lab}: {b:.3f}", (a, b), textcoords="offset points", xytext=(8, -14 if lab == "16" else 4), fontsize=9.5, color=GREEN)
full_line(ax, 26)
ax.set_xlim(0, 26.5); ax.set_ylim(1.22, 1.6)
ax.set_xlabel("tokens read per word, on average (of up to 64)"); ax.set_ylabel("loss (lower is better)")
ax.legend(loc="upper right", fontsize=9.5, ncol=1)
fig.savefig("main_result.png", dpi=170); plt.close(fig)

# 2. how a word's size is chosen: three rules vs a fixed number
fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22)
frame(fig, "Three ways to choose each word's number of tokens give about the same result",
      "Single tokens, sizes rounded up to 4, 8, 16, 32 or 64. Mean of 3 runs; two runs differ by about 0.02–0.04.",
      "top-p: the top tokens that hold a share p of the picker's attention.   ratio: every token scoring at least r times the best token.\n"
      "entropy: c times the picker's effective number of tokens. Below 8 tokens all three beat a fixed number; from 10 tokens on they match it.")
for pts, c, name, labs in ((FIXED, BLUE, "same number for every word (top-k)", ("k=4", "k=8", "k=16", "k=32")),
                           (TOP_P, ORANGE, "top-p", ("p=0.5", "p=0.7", "p=0.8", "p=0.9")),
                           (RATIO, GREEN, "ratio", ("r=0.3", "r=0.2", "r=0.1")),
                           (ENT, PURPLE, "entropy", ("c=0.25", "c=0.5", "c=1", "c=2"))):
    ax.plot([a for a, _ in pts], [b for _, b in pts], color=c, marker="o", ms=6, lw=2, label=name)
    for (a, b), lab in zip(pts, labs):
        off = {"p=0.8": (-40, -4), "c=1": (-6, -15), "k=16": (6, 6), "c=0.5": (6, -12), "k=8": (6, 6)}.get(lab, (-4, -15) if c == GREEN else (5, 5))
        ax.annotate(lab, (a, b), textcoords="offset points", xytext=off, fontsize=8.5, color=c)
full_line(ax, 26)
ax.set_xlim(0, 26.5); ax.set_ylim(1.22, 1.52)
ax.set_xlabel("tokens read per word, on average (of up to 64)"); ax.set_ylabel("loss (lower is better)")
ax.legend(loc="upper right", fontsize=10)
fig.savefig("rules.png", dpi=170); plt.close(fig)

# 3. which sizes are allowed, and how many tokens each task chooses
SIZES = [("4, 8, 16,\n32, 64", 1.339, 5.8, ORANGE), ("any number\nfrom 4", 1.357, 5.3, ORANGE), ("4, 16, 64", 1.324, 7.0, ORANGE),
         ("8, 16,\n32, 64", 1.286, 9.5, ORANGE), ("1, 2, 4, 8,\n16, 32, 64", 1.566, 3.5, GRAY), ("any number\nfrom 1", 1.583, 3.0, GRAY)]
NAME = {"factc": "fact,\ncommon", "factr": "fact,\nrare", "sort": "sort", "chain": "next word\nin text", "copy": "copy a\npiece",
        "count": "count\nzeros", "back2": "repeat\n2 back", "recall": "find a\npartner"}
KEPT = {"factc": 3.0, "factr": 3.0, "sort": 4.2, "chain": 4.9, "copy": 6.2, "count": 8.7, "back2": 9.2, "recall": 17.3}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.24, wspace=0.22)
frame(fig, "The smallest allowed size matters; rounding to fixed sizes costs nothing",
      "top-p with p=0.5, mean of 3 runs. Left: loss for each set of allowed sizes. Right: tokens each task reads with sizes 4 to 64.",
      "Letting a word read only 1 or 2 tokens breaks it (gray). Rounding up to 4, 8, 16, 32 or 64 scores the same as allowing any number from 4.\n"
      "Fact questions are only 3 tokens long, so they read all 3. Finding a word's partner among 30 pairs needs the most tokens.")
ax.bar(range(len(SIZES)), [l for _, l, _, _ in SIZES], color=[c for *_, c in SIZES], width=0.62)
for i, (_, l, k, _) in enumerate(SIZES): ax.text(i, l + 0.006, f"{l:.3f}\n{k} tokens", ha="center", fontsize=9.5, color=INK)
ax.axhline(FULL, color=GRAY, lw=1.5, ls="--")
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([n for n, *_ in SIZES], fontsize=9.5)
ax.set_ylim(1.2, 1.65); ax.set_ylabel("loss (lower is better)")
ax.set_title(f"Allowed sizes (dashed: normal attention {FULL})", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
order = list(KEPT)
bx.bar(range(len(order)), [KEPT[s] for s in order], color=ORANGE, width=0.6)
for i, s in enumerate(order): bx.text(i, KEPT[s] + 0.3, f"{KEPT[s]:.1f}", ha="center", fontsize=10, color=INK)
bx.axhline(8, color=BLUE, lw=1.5, ls=":"); bx.text(-0.3, 8.4, "top-8 reads 8 for every word", ha="left", fontsize=9.5, color=BLUE)
bx.set_xticks(range(len(order))); bx.set_xticklabels([NAME[s] for s in order], fontsize=9.5)
bx.set_ylim(0, 20); bx.set_ylabel("tokens read per word")
bx.set_title("Tokens each task chooses", loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
fig.savefig("sizes.png", dpi=170); plt.close(fig)

# 4. how the picker is trained (fixed top-8)
MODES = [("copy\n(DeepSeek)", 1.314, BLUE), ("copy, over\nall tokens", 1.301, BLUE), ("copy + task\nevery step", 1.394, ORANGE),
         ("copy every\nother step,\ntask always", 1.431, ORANGE), ("copy fades\nout, task\nstays", 1.462, ORANGE),
         ("copy in\nwarm-up,\nthen task", 2.292, ORANGE), ("task only", 2.532, ORANGE)]
fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.28)
frame(fig, "Training the picker to copy the attention beats training it on the task",
      "DeepSeek sparse attention keeping 8 of 64 tokens. Mean of 3 runs.",
      "Copy = the picker learns to match where the main attention looks (KL loss), as in DeepSeek V3.2. Task = the picker's scores are added\n"
      "to the attention scores, so the model's own loss trains it. The more task training, the worse. With task only, every picker score in\n"
      "layer 1 ends at 0 (its ReLU dies), so it keeps random tokens. The picker is trained with Muon in every run.")
ax.bar(range(len(MODES)), [l for _, l, _ in MODES], color=[c for *_, c in MODES], width=0.6, alpha=0.9)
for i, (_, l, _) in enumerate(MODES): ax.text(i, l + 0.03, f"{l:.3f}", ha="center", fontsize=10.5, color=INK)
ax.axhline(FULL, color=GRAY, lw=1.5, ls="--"); ax.text(6.45, FULL - 0.04, f"normal attention {FULL}", ha="right", va="top", fontsize=10, color=INK2)
ax.set_xticks(range(len(MODES))); ax.set_xticklabels([n for n, *_ in MODES], fontsize=10)
ax.set_ylim(1.0, 2.75); ax.set_ylabel("loss (lower is better)")
fig.savefig("picker_training.png", dpi=170); plt.close(fig)

# 5. longer sequences (128 tokens, 32 blocks of 4): a chosen number of blocks vs a fixed number
LONG_FULL = 1.418
LONG_FIXED = [(15.1, 1.643), (28.4, 1.535), (48.8, 1.449)]       # fixed 16, 32, 64 tokens in blocks of 4
LONG_CHOSEN = [(13.6, 1.573), (24.2, 1.458), (32.1, 1.443)]      # p = 0.5, 0.7, 0.85; any whole number of blocks
fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.17)
frame(fig, "Each token picking how many tokens to look at: half as many, same loss",
      "Texts of 128 tokens. Earlier tokens are looked at in groups of 4, the way GPUs load them. Average of 3 runs.",
      "Loss = how wrong the model's next-token guesses are; lower is better.")
ax.plot([a for a, _ in LONG_FIXED], [b for _, b in LONG_FIXED], color=BLUE, marker="o", ms=8, lw=2.5, label="DeepSeek: every token looks at the same number")
ax.plot([a for a, _ in LONG_CHOSEN], [b for _, b in LONG_CHOSEN], color=ORANGE, marker="o", ms=8, lw=2.5, label="each token picks how many")
for a, b in LONG_FIXED: ax.annotate(f"{a:.0f}", (a, b), textcoords="offset points", xytext=(9, 4), fontsize=10.5, color=BLUE)
for a, b in LONG_CHOSEN: ax.annotate(f"{a:.0f}", (a, b), textcoords="offset points", xytext=(-9, -14), ha="right", fontsize=10.5, color=ORANGE)
ax.annotate("", xy=(25.5, 1.452), xytext=(47.3, 1.452), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
ax.text(36.5, 1.462, "half as many tokens,\nsame loss", ha="center", va="bottom", fontsize=11, color=INK)
ax.axhline(LONG_FULL, color=GRAY, lw=1.5, ls="--")
ax.text(65, LONG_FULL - 0.006, "normal attention: looks at all of them", ha="right", va="top", fontsize=10, color=INK2)
ax.set_xlim(0, 66); ax.set_ylim(1.38, 1.68)
ax.set_xlabel("earlier tokens each token looks at, on average"); ax.set_ylabel("loss")
ax.legend(loc="upper right", fontsize=10.5)
fig.savefig("blocks_long.png", dpi=170); plt.close(fig)
# 6. the best minimum grows with the context length
import math
BEST_MIN = [(32, 2), (64, 2), (128, 4), (256, 8), (512, 32)]   # minimum with the lowest loss vs a fixed amount reading as much (RESULTS.md step 8)
fig, ax = plt.subplots(figsize=(11.5, 6.4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.2)
frame(fig, "The best minimum grows with the context length, at about 1/24 of it",
      "Each token chooses how much to read, but never less than the minimum. Single tokens, 2 runs (512 tokens: 1 run for the fixed amounts).",
      "Best = the minimum whose loss is furthest below DeepSeek's fixed amount reading as many tokens. Minimums tried: 1, 2, 4, 8 (and 16, 32, 64\n"
      "at the longest lengths), so each best value is known only to within a factor of 2.")
xs = [a for a, _ in BEST_MIN]
ax.plot(xs, [b for _, b in BEST_MIN], color=ORANGE, marker="o", ms=9, lw=2.5)
for a, b in BEST_MIN: ax.annotate(f"{b} tokens", (a, b), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10.5, color=INK)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.set_yticks([1, 2, 4, 8, 16, 32, 64]); ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_ylim(1, 64); ax.set_xlabel("context length (tokens)"); ax.set_ylabel("best minimum tokens read")
fig.savefig("minimum_scaling.png", dpi=170); plt.close(fig)
# 7. gain over DeepSeek's fixed amount for each minimum, one line per context length (RESULTS.md step 8)
GAIN = {32: [(1, -0.269), (2, 0.029), (4, 0.028), (8, 0.006)],
        64: [(1, -0.088), (2, 0.078), (4, 0.050), (8, 0.019)],
        128: [(1, 0.033), (2, 0.063), (4, 0.089), (8, 0.076)],
        256: [(1, -0.008), (2, 0.090), (4, 0.096), (8, 0.151), (16, 0.028)],
        512: [(8, -0.004), (16, 0.058), (32, 0.294), (64, -0.002)]}
SEQ_COL = {32: "#9ec5ef", 64: "#6aa6e4", 128: "#2a78d6", 256: "#1b559c", 512: "#0d2f5c"}   # light = short, dark = long
fig, ax = plt.subplots(figsize=(11.5, 6.6))
fig.subplots_adjust(left=0.09, right=0.9, top=0.82, bottom=0.2)
frame(fig, "Longer texts need a larger minimum: the best one moves right as the context grows",
      "How much lower the loss is than DeepSeek's fixed amount reading as many tokens. Higher is better. Big dot = best minimum.",
      "Each token chooses how much to read, but never less than the minimum. Single tokens, 2 runs (512 tokens: 1 run for the fixed amounts).\n"
      "Below 0 = worse than DeepSeek's fixed amount: a too-small minimum lets tokens miss the word they need; a too-large one wastes reading.")
ax.axhline(0, color=GRAY, lw=1.5, ls="--")
ax.text(17, -0.015, "same as DeepSeek's fixed amount", ha="left", va="top", fontsize=10, color=INK2)
for L, pts in GAIN.items():
    c = SEQ_COL[L]; xs, ys = [a for a, _ in pts], [b for _, b in pts]
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=6)
    bx, by = max(pts, key=lambda q: q[1]); ax.plot(bx, by, "o", ms=13, color=c, mec="white", mew=2)
    i, dy = (0, -9) if L in (32, 64) else (-1, -12 if L == 512 else 0)   # 32 and 64 labelled at their left end, clear of the others
    ax.annotate(f"{L} tokens", (xs[i], ys[i]), textcoords="offset points", xytext=(9, dy), va="center", fontsize=10.5, color=INK)
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_xlim(0.8, 100); ax.set_ylim(-0.3, 0.33)
ax.set_xlabel("minimum tokens each token must read"); ax.set_ylabel("loss below DeepSeek's fixed amount")
fig.savefig("minimum_by_length.png", dpi=170); plt.close(fig)
# 8. calculated attention compute in DeepSeek-V3.2 (config.json sizes), if each token reads half of DeepSeek's 2,048 on average
MAIN = 2 * 128 * (576 + 512)   # operations per earlier token read: 128 heads, 576-wide keys (512 latent + 64 position), 512-wide values
PICK = 2 * 64 * 128            # operations per earlier token scored by the picker: 64 heads of width 128
CTX = [8192, 16384, 32768, 65536, 131072]
fig, ax = plt.subplots(figsize=(11.5, 6.6))
fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2)
frame(fig, "Halving the reading saves 40% of attention compute at 8K tokens, 10% at 128K",
      "Calculated, not measured: DeepSeek-V3.2 attention, one layer, one new token. Assumes each token reads 1,024 on average instead of 2,048.",
      "The picker must still score every earlier token, so its cost (gray) grows with the text and is the same in both versions; only the reading (color) halves.\n"
      "Normal attention, which reads every token, would cost 2.3 (8K) to 36.5 (128K) billion operations. Counts operations, not time: the picker runs in FP8.")
w = 0.36
for i, L in enumerate(CTX):
    pick = PICK * L / 1e9
    for j, (k, c) in enumerate([(2048, BLUE), (1024, ORANGE)]):
        x = i + (j - 0.5) * (w + 0.03); read = MAIN * k / 1e9
        ax.bar(x, pick, w, color=GRAY, edgecolor="white", linewidth=2)
        ax.bar(x, read, w, bottom=pick, color=c, edgecolor="white", linewidth=2)
        ax.text(x, pick + read + 0.04, f"{pick + read:.2f}", ha="center", fontsize=10, color=INK)
    full, half = PICK * L + MAIN * 2048, PICK * L + MAIN * 1024
    ax.text(i + 0.5 * (w + 0.03), (half / 1e9) + 0.22, f"−{1 - half / full:.0%}", ha="center", fontsize=11.5, fontweight="bold", color=ORANGE)
ax.set_xticks(range(len(CTX))); ax.set_xticklabels(["8K", "16K", "32K", "64K", "128K"])
ax.bar(0, 0, color=BLUE, label="reading 2,048 tokens (DeepSeek)"); ax.bar(0, 0, color=ORANGE, label="reading 1,024 on average (each token chooses)")
ax.bar(0, 0, color=GRAY, label="picker scoring every earlier token")
ax.legend(loc="upper left", fontsize=10.5); ax.set_ylim(0, 3.2); ax.grid(axis="x", visible=False)
ax.set_xlabel("text length (tokens)"); ax.set_ylabel("billion operations per new token, per layer")
fig.savefig("compute_saving.png", dpi=170); plt.close(fig)
print("wrote main_result.png rules.png sizes.png picker_training.png blocks_long.png minimum_scaling.png minimum_by_length.png compute_saving.png")
