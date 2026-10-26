#!/bin/bash
# Train pix2point on AASCE, then run inference + evaluation on the test set
# with the best-validation-MRE checkpoint.
#
#   from scratch:
#     sbatch --partition=<gpu_partition> slurm/train_aasce.slurm
#   from a pretrained backbone (only the backbone initialisation differs):
#     sbatch --partition=<gpu_partition> \
#       --export=ALL,PRETRAINED_BACKBONE=/path/to/pretrained_backbone.pth slurm/train_aasce.slurm
#
# Submit from the repository root. Resources of the reference runs: one 40 GB
# GPU (batch size 8); early stopping on the validation error (patience 40 epochs).
#SBATCH --job-name=pix2point_aasce
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=60GB
#SBATCH --time=05:00:00

set -euo pipefail

# ----- Environment: adapt to your cluster ---------------------------------------
# Activate the Python environment holding requirements.txt, e.g.:
#   source /path/to/venv/bin/activate      or      conda activate pix2point
DATA_ROOT=${DATA_ROOT:-data/aasce}                 # see README, "Data"
PRETRAINED_BACKBONE=${PRETRAINED_BACKBONE:-}       # empty -> from scratch
SEED=${SEED:-42}
INIT=$([ -n "$PRETRAINED_BACKBONE" ] && echo pretrained || echo scratch)
OUTPUT_DIR=${OUTPUT_DIR:-runs/aasce_${INIT}_seed${SEED}_job${SLURM_JOB_ID:-local}}

# ----- Computation ------------------------------------------------------------
python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -u train.py --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --seed "$SEED" \
    ${PRETRAINED_BACKBONE:+--pretrained-backbone "$PRETRAINED_BACKBONE"}

python -u predict.py --data-root "$DATA_ROOT" --weights-dir "$OUTPUT_DIR/train" \
    --checkpoint best_mre --output-dir "$OUTPUT_DIR/test_best_mre"
