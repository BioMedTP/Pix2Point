#!/bin/bash
# Self-supervised pretraining of the pix2point backbone on AASCE.
#
#   sbatch --partition=<gpu_partition> slurm/pretrain_aasce.slurm
#
# Submit from the repository root. The reference run took ~2 h 25 min on one
# 48 GB GPU (200 epochs, batch size 4). Then train from its last-epoch backbone:
#   sbatch --partition=<gpu_partition> \
#     --export=ALL,PRETRAINED_BACKBONE=<OUTPUT_DIR>/train/unet_backbone_last.pth slurm/train_aasce.slurm
#SBATCH --job-name=pix2point_aasce_pretrain
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
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-runs/aasce_pretraining_seed${SEED}_job${SLURM_JOB_ID:-local}}

# ----- Computation ------------------------------------------------------------
python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -u pretrain.py --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --seed "$SEED"
