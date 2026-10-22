#!/bin/zsh
# one run: 2-layer MoE toy, sequences of 64 tokens, DeepSeek sparse attention in both layers, 260 steps, Muon.
# any knob can be overridden from the environment.  usage: env [knobs] ./run.sh SEED   (appends a row to sampled_runs.jsonl)
cd ${0:A:h}
export MSIGN=${MSIGN:-ns} MOE=${MOE:-8} EMB_MULT=${EMB_MULT:-8} HEAD_MULT=${HEAD_MULT:-8} EMB_B1=${EMB_B1:-0.5} EXP_MOM=${EXP_MOM:-0.8} EMB_WARM=${EMB_WARM:-1} WARMUP=${WARMUP:-1} ATTN_GATE=${ATTN_GATE:-1} HEADS=${HEADS:-2} VE=${VE:-1} VE_MULT=${VE_MULT:-32} CANON=${CANON:-1} CANON_K=${CANON_K:-2} CANON_MULT=${CANON_MULT:-16} CANON_POS=${CANON_POS:-k} CANON_LAYERS=${CANON_LAYERS:-1}
export SEQ=${SEQ:-64} DSA_K=${DSA_K:-64} DSA_KC=${DSA_KC:-1} DSA_WARM=${DSA_WARM:-60} DSA_MUON=${DSA_MUON:-1}
nice -n 20 python3 sampled.py full $1 ${STEPS:-260} ${LR:-0.028} 2>&1 | grep -v Warning
