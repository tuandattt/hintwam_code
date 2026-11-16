# HINT-WAM: Hindsight-Informed Training for World Action Models

Training code for HINT-WAM, a post-training method for the Fast-WAM world action model.

During post-training, the same model plays two roles:
- **Teacher:** additionally conditioned on *hindsight*, the future video of the demonstration.
- **Student:** sees only the current observation.

The student samples action chunks with its own sampler. These samples are matched to the
teacher's action with the energy score, plus a behaviour-cloning anchor on the demonstrations.

This repository contains the training code only. Closed-loop evaluation uses the LIBERO and
RoboTwin evaluators of [Fast-WAM](https://arxiv.org/abs/2603.16666) ([project page](https://yuantianyuan01.github.io/FastWAM/)).

## Contents

```text
hintwam_released/
├── configs/
│   ├── train.yaml                 # global training defaults
│   ├── model/fastwam.yaml         # Fast-WAM (MoT: frozen 5B video expert + 1B action expert)
│   ├── data/                      # LIBERO-90 and RoboTwin 2.0 datasets
│   └── task/                      # one config per training stage (see below)
├── scripts/
│   ├── train.py                   # Hydra entry point
│   ├── train_zero2.sh             # multi-GPU launcher (DeepSpeed ZeRO-2), used for all runs
│   ├── train_zero1.sh             # ZeRO-1 launcher
│   ├── precompute_text_embeds.py  # caches T5 instruction embeddings before training
│   ├── accelerate_configs/  ds_configs/
└── src/fastwam/
    ├── models/wan22/              # Fast-WAM model, gated hindsight adapters (phi_adapter.py)
    ├── opsd/                      # HINT-WAM: teacher/student engine, energy score, sampler rollout
    ├── datasets/                  # LeRobot-format video/action datasets
    └── trainer.py, runtime.py
```

The training stages and their configs:

| Stage | LIBERO-90 config | RoboTwin 2.0 config | Starts from |
|---|---|---|---|
| 0. SFT | `libero90_sft` | (not needed) | released Fast-WAM checkpoint |
| A. Hindsight teacher | `libero90_opsd_teacher` | `robotwin_hs_teacher` | SFT (LIBERO) / released checkpoint (RoboTwin) |
| B. **HINT-WAM** | `libero90_opsd_distill_energy` | `robotwin_hs_distill_energy` | same as Stage A, plus the Stage A teacher |
| BC control | `libero90_bc` | `robotwin_hs_bc` | same as Stage A |
| Ablations | `libero90_opsd_teacher_shuffled`, `libero90_opsd_distill_mse` | `robotwin_hs_teacher_shuffled` | see below |

The released RoboTwin checkpoint was already trained on the RoboTwin data, so it takes the role of
SFT there.

## 1. Environment

Tested with Python 3.10, CUDA 12.8, PyTorch 2.7.1, on 8 × 24 GB GPUs.

```bash
conda create -n hintwam python=3.10 -y
conda activate hintwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Optional settings:

```bash
export RUNS_ROOT=$PWD/runs                                # where runs are written (default ./runs)
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints         # where Wan2.2 weights are cached
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 2. Pretrained checkpoints

**Fast-WAM checkpoints and normalization statistics.** Every stage starts from these, and the
data configs read the statistics from this exact location:

```bash
pip install -U huggingface_hub
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

**Wan2.2 components.** The Wan2.2-TI2V-5B VAE, the T5 text encoder and the tokenizer are
downloaded automatically from ModelScope on first use, into `$DIFFSYNTH_MODEL_BASE_PATH`
(default `./checkpoints`).

## 3. Datasets

### LIBERO-90

These are the LIBERO-90 human demonstrations in LeRobot format, with no-op steps removed:
90 tasks, 3,921 episodes. 10% of the episodes are held out automatically for validation losses.

```bash
huggingface-cli download IPEC-COMMUNITY/libero_90_no_noops_lerobot --repo-type dataset \
  --local-dir ./data/libero/libero_90_no_noops_lerobot
```

### RoboTwin 2.0

This is the preprocessed RoboTwin 2.0 data released with Fast-WAM:

```bash
mkdir -p data/robotwin2.0 && cd data/robotwin2.0
huggingface-cli download yuanty/robotwin2.0-fastwam --repo-type dataset --local-dir .
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
# The data config reads data/robotwin2.0/dataset_stats.json. If the archive has none, use the released statistics:
[ -f data/robotwin2.0/dataset_stats.json ] || \
  cp checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json data/robotwin2.0/dataset_stats.json
```

```text
data/
├── libero/libero_90_no_noops_lerobot/   # meta/  data/  videos/
└── robotwin2.0/
    ├── dataset_stats.json
    └── robotwin2.0/                     # meta/  data/  videos/
```

If your data is elsewhere, either edit `configs/data/*.yaml` or pass the paths as overrides:
`data.train.dataset_dirs=[/path] data.val.dataset_dirs=[/path]`.

### Text-embedding cache (once per dataset)

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero90_sft
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=robotwin_hs_teacher
```

This writes `data/text_embeds_cache/libero90` and `data/text_embeds_cache/robotwin`.

## 4. Training

Usage: `bash scripts/train_zero2.sh <num_gpus> task=<config> [overrides]`. Runs are written to
`$RUNS_ROOT/<config>/<date_time>/`, and weights to `checkpoints/weights/step_XXXXXX.pt`.

The configs hold the paper's settings:

| | SFT | Stage A (teacher) | Stage B (HINT-WAM) |
|---|---|---|---|
| Trainable | action expert | hindsight adapters + latent provider (~13M) | action expert |
| Learning rate (cosine) | 1e-4 | 3e-4 | 1e-4 |
| Steps | 5,000 (step 3,000 is used) | 2,000 | 2,000 |
| Effective batch | 192 (4 × 8 GPUs × 6) | 256 (8 × 8 GPUs × 4) | 192 (4 × 8 GPUs × 6) |

The video expert is frozen in every stage. Stage B also uses:
- M = 4 student samples;
- 10 sampler steps;
- loss weights λ_distill = λ_bc = 1.

**On a different number of GPUs, keep the effective batch the same.** For example, on 4 GPUs pass
`gradient_accumulation_steps=12`. If you run out of memory, pass
`batch_size=2 gradient_accumulation_steps=12` (on 8 GPUs).

### LIBERO-90

```bash
export BASE=checkpoints/fastwam_release/libero_uncond_2cam224.pt

# Stage 0: SFT on LIBERO-90
bash scripts/train_zero2.sh 8 task=libero90_sft resume=$BASE
export SFT=$RUNS_ROOT/libero90_sft/<run>/checkpoints/weights/step_003000.pt

# Stage A: hindsight teacher (only the adapters are saved, ~26 MB per checkpoint)
bash scripts/train_zero2.sh 8 task=libero90_opsd_teacher resume=$SFT
export TEACHER=$RUNS_ROOT/libero90_opsd_teacher/<run>/checkpoints/weights/step_002000.pt

# Stage B: HINT-WAM
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=$TEACHER
```

**Checks during training:**
- **Stage A:** `phi_gate` in the log starts at 0 and should rise. If it stays near 0, the teacher
  is not using hindsight, and Stage B has nothing to learn from.
- **Stage B:** the first log lines must show `teacher loaded from <path>: N adapter + M provider tensors`.
  Without this line, the teacher is untrained and the run is void.

The final model is `$RUNS_ROOT/libero90_opsd_distill_energy/<run>/checkpoints/weights/step_002000.pt`.

### RoboTwin 2.0

```bash
export BASE_R=checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt

bash scripts/train_zero2.sh 8 task=robotwin_hs_teacher resume=$BASE_R
export TEACHER_R=$RUNS_ROOT/robotwin_hs_teacher/<run>/checkpoints/weights/step_002000.pt

bash scripts/train_zero2.sh 8 task=robotwin_hs_distill_energy resume=$BASE_R opsd.teacher_ckpt=$TEACHER_R
```

### Baselines and ablations (LIBERO-90)

Each starts from the same SFT checkpoint and uses the same budget as HINT-WAM:

```bash
# BC control: continue SFT with behaviour cloning only, no teacher
bash scripts/train_zero2.sh 8 task=libero90_bc resume=$SFT

# Objective: velocity matching (MSE), or action-space MSE, instead of the energy score
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_mse resume=$SFT opsd.teacher_ckpt=$TEACHER
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=$TEACHER opsd.loss_type=action_mse

# Off-policy states: supervise on noised demonstration actions instead of the student's own samples
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=$TEACHER opsd.state_source=demo

# Number of student samples M (default 4), and no BC anchor
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=$TEACHER opsd.energy_samples=2
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=$TEACHER opsd.lambda_bc=0

# Shuffled hindsight: a teacher that reads another episode's future, then Stage B with it
bash scripts/train_zero2.sh 8 task=libero90_opsd_teacher_shuffled resume=$SFT
bash scripts/train_zero2.sh 8 task=libero90_opsd_distill_energy resume=$SFT opsd.teacher_ckpt=<shuffled teacher step_002000.pt>
```

## 5. Evaluation

The trained `step_XXXXXX.pt` files load like any Fast-WAM checkpoint. Evaluate them with the
LIBERO and RoboTwin evaluators of the Fast-WAM repository, using:
- the configs `libero_uncond_2cam224_1e-4` and `robotwin_uncond_3cam_384_1e-4`;
- the released normalization statistics;
- 10 sampler steps.

## Acknowledgements

This code builds on [Fast-WAM](https://arxiv.org/abs/2603.16666) ([project page](https://yuantianyuan01.github.io/FastWAM/)) (MIT License), which
in turn uses Wan2.2 and DiffSynth-Studio components. The LIBERO-90 data comes from
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), in the LeRobot conversion by
IPEC-COMMUNITY. The RoboTwin data comes from [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin).

```bibtex
@article{yuan2026fastwam,
  title   = {Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author  = {Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal = {arXiv preprint arXiv:2603.16666},
  year    = {2026}
}
```
