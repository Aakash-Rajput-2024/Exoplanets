#!/usr/bin/env bash
# =============================================================================
# Local (Apple Silicon / MPS) driver for the multi-seed study.
#
# Mac counterpart to compute/submit_remaining.sh. One MPS device means runs go
# STRICTLY SERIALLY -- there is no concurrency to exploit, and overlapping two
# runs on 18 GB of unified memory only causes paging.
#
# RESUMABILITY
#   Every run resumes, including MID-EPOCH. train_runner writes a full snapshot
#   (model + optimiser + scheduler + RNG + loader shuffle position + batch index)
#   every --save-every seconds, so a Ctrl-C, a crash, or a closed lid costs at
#   most that many seconds of compute -- not the whole epoch. Re-run this script
#   and it picks up exactly where it stopped.
#
# COST on this machine (M3 Pro, 18 GB), measured from the repo's own logs:
#   original1dcnn ~124 s/ep (~1.7 h/seed) | transformerarch ~941 s/ep (~13.1 h)
#   causal ~990 s/ep (~13.8 h)            | causal_cfi ~990 s/ep (~13.8 h)
#   All nine pending runs is ~79 h of wall clock. Run a subset.
#
# USAGE
#   ./train.sh --list                  what is pending, cheapest track first
#   ./train.sh --only original1dcnn    one track, all its pending seeds (~5 h)
#   ./train.sh --budget 6              start no new run once 6 h have elapsed
#   ./train.sh --save-every 120        snapshot every 2 min instead of 5
#   ./train.sh                         everything pending (~79 h)
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src"

PY="${PY:-python}"
ONLY=""
BUDGET_H=0
LIST_ONLY=0
EPOCHS="${EPOCHS:-50}"
SAVE_EVERY="${SAVE_EVERY:-300}"
RETRIES="${RETRIES:-4}"   # auto-resume attempts per run after a crash
YES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --list)       LIST_ONLY=1 ;;
        --only)       ONLY="$2"; shift ;;
        --budget)     BUDGET_H="$2"; shift ;;
        --epochs)     EPOCHS="$2"; shift ;;
        --save-every) SAVE_EVERY="$2"; shift ;;
        --retries)    RETRIES="$2"; shift ;;
        -y|--yes)     YES=1 ;;
        -h|--help)    sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---- preflight ---------------------------------------------------------------
if [ ! -f data/cache_v2/manifest.json ]; then
    echo "ERROR: data/cache_v2/manifest.json missing. Build it once with:" >&2
    echo "    PYTHONPATH=src $PY -m common.build_cache" >&2
    exit 1
fi

# Resume checkpoints are pickles carrying numpy RNG state, so the interpreter
# that reads them must have a numpy ABI-compatible with the one that wrote them.
# numpy 2.x pickles reference `numpy._core`, which does not exist in 1.x - loading
# then dies at torch.load AFTER the model is built, which looks like the run
# started when it never did. Prefer the project venv unless told otherwise.
VENV="/Users/aakashrajput/MachineLearning/venvNPT"
if [ -z "${PY_EXPLICIT:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
    echo "note: no venv active; using $VENV/bin/python (set PY_EXPLICIT=1 to override)"
fi

$PY -c 'import sys, torch, numpy
ok = torch.backends.mps.is_available()
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | numpy {numpy.__version__} | mps {ok}")
if not ok:
    print("WARNING: MPS unavailable - training falls back to CPU and will be "
          "several times slower than the estimates below.")' || exit 1

# Fail fast if this interpreter cannot read the snapshots it would need to resume.
CK=$(ls -t src/models/*/checkpoints_v2/last_*.pth \
        src/models/causal/cnn_trnas/checkpoints_v2/last_*.pth 2>/dev/null | head -1)
if [ -n "$CK" ]; then
    if ! $PY -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$CK" 2>/dev/null; then
        echo "ERROR: this python cannot load existing resume checkpoints." >&2
        echo "       tried: $CK" >&2
        echo "       Usually a numpy major-version mismatch against the interpreter" >&2
        echo "       that wrote them. Activate the project venv and retry:" >&2
        echo "           source $VENV/bin/activate" >&2
        exit 1
    fi
fi

# Regenerate the plan from what is actually on disk, priced for THIS machine, so
# the hour estimates below are Mac hours rather than cluster hours.
$PY compute/seed_plan.py --device mac --emit-manifest >/dev/null || exit 1
MANIFEST="compute/remaining_runs.tsv"
[ -s "$MANIFEST" ] || { echo "ERROR: empty $MANIFEST" >&2; exit 1; }

# ---- build the run list ------------------------------------------------------
# Cheapest track first, which here is also the right scientific order:
# original1dcnn has the joint-highest ceiling (0.779) at a fraction of the cost,
# and its current number comes from a 10-epoch run - the cheapest defect to fix.
ORDER="original1dcnn optimized1dcnn transformerarch causal causal_cfi causal_xl"

RUNS_FILE="$(mktemp -t seedruns)"
trap 'rm -f "$RUNS_FILE"' EXIT
for track in $ORDER; do
    while IFS=$'\t' read -r t seed ep cloud flags est st; do
        case "$t" in \#*|"") continue ;; esac
        [ "$t" = "$track" ] || continue
        [ -n "$ONLY" ] && [ "$t" != "$ONLY" ] && continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$seed" "$cloud" "$flags" "$est" "$st" >> "$RUNS_FILE"
    done < "$MANIFEST"
done

N=$(wc -l < "$RUNS_FILE" | tr -d ' ')
if [ "$N" -eq 0 ]; then
    echo "Nothing pending${ONLY:+ for track '$ONLY'}."
    exit 0
fi

echo
echo "=================================================================="
echo " Local training plan (Apple Silicon, serial, mid-epoch resumable)"
echo "=================================================================="
printf " %-3s %-16s %-4s %-10s %8s  %s\n" "#" "track" "seed" "state" "est_h" "variant"
i=0; TOTAL_EST=0
while IFS=$'\t' read -r t seed cloud flags est st; do
    i=$((i + 1))
    TOTAL_EST=$(awk -v a="$TOTAL_EST" -v b="$est" 'BEGIN{print a+b}')
    [ "$cloud" = "-" ] && cloud=""
    printf " %-3d %-16s %-4s %-10s %8.1f  %s\n" "$i" "$t" "$seed" "$st" "$est" "$cloud"
done < "$RUNS_FILE"
awk -v h="$TOTAL_EST" 'BEGIN{printf "\n total estimated: %.1f h (%.1f days continuous)\n", h, h/24}'
[ "$BUDGET_H" != "0" ] && echo " budget cap     : ${BUDGET_H} h (no NEW run starts past this)"
echo " snapshot every : ${SAVE_EVERY}s (max work lost to an interrupt)"
echo " auto-retry     : ${RETRIES} attempts per run (MPS crashes are expected here)"
echo "=================================================================="

[ "$LIST_ONLY" -eq 1 ] && exit 0

if [ "$YES" -eq 0 ] && [ -t 0 ]; then
    printf "Start? [y/N] "
    read -r ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUNLOG="$REPO/logs/local_$TS"
mkdir -p "$RUNLOG"
{
    echo "timestamp: $TS"
    echo "host: $(hostname)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "memory_gb: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty: $(git diff --quiet 2>/dev/null && echo no || echo yes)"
    echo "epochs: $EPOCHS"
    echo "save_every_s: $SAVE_EVERY"
} > "$RUNLOG/provenance.txt"
cp "$MANIFEST" "$RUNLOG/manifest.tsv"

# Ctrl-C must stop the DRIVER, not just the current python process - otherwise
# the trap kills one run and the loop cheerfully starts the next.
STOPPED=0
trap 'STOPPED=1; echo ""; echo "[interrupt] stopping after this run snapshots; rerun to resume."' INT TERM

START=$(date +%s)
i=0; OK=0; FAILED=0
while IFS=$'\t' read -r t seed cloud flags est st; do
    [ "$STOPPED" -eq 1 ] && break
    i=$((i + 1))

    ELAPSED_H=$(awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}')
    if [ "$BUDGET_H" != "0" ]; then
        OVER=$(awk -v e="$ELAPSED_H" -v x="$est" -v b="$BUDGET_H" 'BEGIN{print (e+x > b) ? 1 : 0}')
        if [ "$OVER" -eq 1 ]; then
            echo
            echo "[budget] ${ELAPSED_H}h used; next run needs ~${est}h, over the ${BUDGET_H}h cap. Stopping."
            break
        fi
    fi

    EXTRA=""
    [ "$flags" != "-" ] && EXTRA="$flags"
    [ "$cloud" != "-" ] && EXTRA="$EXTRA --cloud $cloud"
    TAG="${t}_s${seed}"; [ "$cloud" != "-" ] && TAG="${TAG}_${cloud}"

    echo
    echo "---------------------------------------------------------------"
    printf " [%d/%d] %s seed %s  (%s)  est %sh  start %s\n" \
           "$i" "$N" "$t" "$seed" "$st" "$est" "$(date '+%H:%M')"
    echo "---------------------------------------------------------------"

    RS=$(date +%s)
    # The transformer tracks die on MPS - SIGKILL (137, memory reaper) and
    # SIGSEGV (139) have both been observed, while every CNN run finished clean.
    # Each attempt still banks epochs via --resume, so retrying converts a crash
    # from "restart by hand" into "lose the last --save-every seconds". We stop
    # retrying when an attempt makes NO progress, which distinguishes a flaky
    # crash from a hard failure that would otherwise loop forever.
    ATTEMPT=0
    while :; do
        ATTEMPT=$((ATTEMPT + 1))
        BEFORE=$($PY compute/seed_plan.py --device mac --progress-of "$t:$seed" 2>/dev/null || echo 0)

        [ "$ATTEMPT" -gt 1 ] && echo " retry $((ATTEMPT-1))/$RETRIES for $TAG (resuming from epoch $BEFORE)"
        caffeinate -is $PY -m common.train_runner "$t" \
            --seed "$seed" --epochs "$EPOCHS" --resume --diag \
            --save-every "$SAVE_EVERY" $EXTRA 2>&1 | tee -a "$RUNLOG/${TAG}.log"
        RC=${PIPESTATUS[0]}
        [ "$RC" -eq 0 ] && break
        [ "$STOPPED" -eq 1 ] && break
        [ "$ATTEMPT" -gt "$RETRIES" ] && break

        AFTER=$($PY compute/seed_plan.py --device mac --progress-of "$t:$seed" 2>/dev/null || echo 0)
        if [ "$AFTER" -le "$BEFORE" ]; then
            echo " WARN: $TAG exited $RC with no progress ($BEFORE->$AFTER epochs); not retrying" >&2
            break
        fi
        echo " $TAG crashed (exit $RC) at epoch $AFTER - retrying in 30s" >&2
        sleep 30
    done
    RH=$(awk -v a="$RS" -v b="$(date +%s)" 'BEGIN{printf "%.2f", (b-a)/3600}')
    if [ "$RC" -eq 0 ]; then
        echo " done $TAG in ${RH}h ($ATTEMPT attempt(s))"
        OK=$((OK + 1))
    else
        echo " WARN: $TAG still incomplete after ${RH}h / $ATTEMPT attempt(s) - rerun to resume" >&2
        FAILED=$((FAILED + 1))
    fi
done < "$RUNS_FILE"

TOTAL_H=$(awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}')
echo
echo "Ran $i of $N in ${TOTAL_H}h  ($OK completed, $FAILED still incomplete). Logs: $RUNLOG"
echo
echo "Next:"
echo "  ./train.sh --list                              # what is still pending"
echo "  PYTHONPATH=src $PY -m evaluation.run_eval      # re-evaluate"
echo "  PYTHONPATH=src $PY -m common.summarize         # mean±std table"
