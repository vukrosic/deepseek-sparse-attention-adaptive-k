#!/bin/zsh
# blocks of 4 tokens on longer texts: each token chooses its number of blocks vs the same number for every token.
# usage: ./blocks_sweep.sh SEQ SEED [chosen p values] [fixed sizes]   ("full" in fixed sizes = normal attention, "none" = skip)
cd ${0:A:h}
L=$1; S=$2; PS=(${=3:-0.7 0.85}); FIX=(${=4:-32 64 128}); [[ $3 == none ]] && PS=(); [[ $4 == none ]] && FIX=()
b=4; s=4; while (( s + 4 <= L )); do s=$((s + 4)); b="$b,$s"; done   # any whole number of blocks
for p in $PS; do
  t0=$SECONDS; env SEQ=$L DSA_K=$L DSA_BLK=4 DSA_P=$p DSA_BUCKETS=$b ./run.sh $S | grep -v kept; echo "  L=$L chosen p=$p seed=$S $((SECONDS - t0))s"
done
for k in $FIX; do
  t0=$SECONDS
  if [[ $k == full ]]; then env SEQ=$L DSA_K=$L ./run.sh $S | grep -v kept
  else env SEQ=$L DSA_K=$L DSA_BLK=4 DSA_P=0.5 DSA_BUCKETS=$k ./run.sh $S | grep -v kept; fi
  echo "  L=$L fixed=$k seed=$S $((SECONDS - t0))s"
done
