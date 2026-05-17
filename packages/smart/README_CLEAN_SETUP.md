Activate UV environment and install packages
Navigate to packages/smart

Run this:
bash scripts/setup_av2.sh \
    --dev \
    --scenarios 10000 \
    --create-splits \
    --train-size 8000 \
    --val-size 2000 \
    --split-seed 42 \
    --train-split-out datasets/train \
    --val-split-out datasets/val \
    --hash-file-contents

Run this:
python train.py \
    --lr 1.1103e-3 \
    --dropout 0.1926 \
    --hidden_dim 256 \
    --save_ckpt_path checkpoints/run_01 \
    --return_metrics metrics/run_01.json \
    --max_epochs 100 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --train_path train_split \
    --val_path validation_split \
    --devices 2 \
    --num_cpu_workers 8 \
    --validate_dataset
    --remove_bad_files