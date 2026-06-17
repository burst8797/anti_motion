# AntiMotion

AntiMotion generates adversarial perturbations for videos before training MotionDirector Temporal LoRA. The repository is organized as a lightweight code release: it includes the protection script and the minimal MotionDirector model and LoRA components needed to run it, but excludes model weights, generated videos, evaluation outputs, and experiment artifacts.

## Structure

```text
AntiMotion/
├── anti_motion.py
├── model_loader.py
├── third_party/
│   └── motiondirector/
│       ├── models/
│       │   ├── unet_3d_blocks.py
│       │   └── unet_3d_condition.py
│       └── utils/
│           ├── convert_diffusers_to_original_ms_text_to_video.py
│           ├── lora.py
│           └── lora_handler.py
├── scripts/
│   └── run_example.sh
├── requirements.txt
├── LICENSE
└── README.md
```

`anti_motion.py` and `model_loader.py` are the project code. `third_party/motiondirector/` contains the minimal MotionDirector-compatible UNet and LoRA utilities required to run the protection script.

## Setup

Create a clean Python environment. Python 3.10 or 3.11 is recommended.

```bash
conda create -n antimotion python=3.11 -y
conda activate antimotion

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, install PyTorch first with the matching index, then install the remaining requirements:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Model Weights

Download a text-to-video base model separately. The weights are intentionally not tracked in this repository.

```bash
hf download cerspense/zeroscope_v2_576w \
  --local-dir ./zeroscope_v2_576w \
  --max-workers 8
```

ModelScope T2V can also be used:

```bash
hf download ali-vilab/text-to-video-ms-1.7b \
  --local-dir ./modelscope_t2v_1.7b \
  --max-workers 8
```

## Usage

Run on a single video:

```bash
CUDA_VISIBLE_DEVICES=0,1 python anti_motion.py \
  --video_path "/path/to/input.mp4" \
  --output_path "./protected_videos/protected_input.mp4" \
  --model_path "./zeroscope_v2_576w" \
  --prompt "a person is moving" \
  --epsilon 0.05 \
  --pgd_steps 20 \
  --chunk_size 5 \
  --inner_steps 5 \
  --chaos_weight 0.2 \
  --global_weight 0.5
```

Run on every `.mp4` file in a folder:

```bash
CUDA_VISIBLE_DEVICES=0,1 python anti_motion.py \
  --video_path "/path/to/video_folder" \
  --output_path "./protected_videos" \
  --model_path "./zeroscope_v2_576w" \
  --prompt "a person is moving"
```

The output path is a file when `--video_path` is a single video, and a folder when `--video_path` is a directory.

## Main Arguments

| Argument | Description |
| --- | --- |
| `--video_path` | Input `.mp4` file or folder containing `.mp4` files. |
| `--output_path` | Output file or output folder. |
| `--model_path` | Local path to the base text-to-video model. |
| `--prompt` | Text prompt used when computing the protection objective. |
| `--epsilon` | Perturbation budget in `[0, 1]` pixel space. |
| `--pgd_steps` | Number of per-chunk PGD steps. |
| `--chunk_size` | Number of frames per chunk. |
| `--inner_steps` | LoRA inner update steps. |
| `--chaos_weight` | Weight for intra-chunk temporal discontinuity. |
| `--global_weight` | Weight for cross-chunk boundary discontinuity. |
| `--global_pgd_steps` | Additional cross-chunk PGD steps. Set to `0` to disable. |
| `--no_transform` | Disable transform sampling. |

## What Is Not Included

The repository intentionally ignores:

- base model weights
- trained LoRA checkpoints
- generated or protected videos
- evaluation outputs
- local datasets
- caches and environment folders

This keeps the GitHub release focused on code only.

## Acknowledgements

This code uses MotionDirector-compatible model and LoRA components under `third_party/motiondirector/`. Please also follow the upstream licenses for MotionDirector, ModelScope, Hugging Face Diffusers, and any base model weights you use.
