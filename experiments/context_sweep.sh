#!/bin/zsh
# minimum tokens read vs context length.  usage: ./context_sweep.sh SEQ SEED [minimums] [fixed sizes]
# chosen size: p=0.5, allowed sizes = the minimum, then powers of 2 up to SEQ.  fixed size: every token reads the same k tokens.
cd ${0:A:h}
L=$1; S=$2; MINS=(${=3:-1 2 4 8}); FIX=(${=4:-4 8 16})
for m in $MINS; do
  b=$m; s=$m; while (( s * 2 <= L )); do s=$((s * 2)); b="$b,$s"; done
  t0=$SECONDS; env SEQ=$L DSA_K=$L DSA_P=0.5 DSA_BUCKETS=$b ./run.sh $S | grep -v kept; echo "  L=$L min=$m seed=$S $((SECONDS - t0))s"
done
for k in $FIX; do
  t0=$SECONDS; env SEQ=$L DSA_K=$k ./run.sh $S | grep -v kept; echo "  L=$L fixed=$k seed=$S $((SECONDS - t0))s"
done
