<p align="center">
  <h1 align="center">AntiMotion</h1>
  <p align="center">
    Adversarial video perturbation for disrupting MotionDirector Temporal LoRA training.
    <br>
    <a href="#overview">Overview</a> |
    <a href="#results-and-video-demo">Demo</a> |
    <a href="#installation">Installation</a> |
    <a href="#usage">Usage</a> |
    <a href="#acknowledgements">Acknowledgements</a>
  </p>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

## Overview

AntiMotion is a lightweight research codebase for generating protected videos before MotionDirector-style Temporal LoRA training. Given an input video and prompt, AntiMotion optimizes a bounded perturbation that targets temporal adaptation in text-to-video diffusion models while keeping the output in the original video format.

The current release includes:

- `AntiMotion/anti_motion.py`: two-stage PGD protection for single videos or folders of `.mp4` files.
- `AntiMotion/model_loader.py`: Diffusers/Transformers model loading helpers.
- `AntiMotion/third_party/motiondirector/`: the minimal MotionDirector-compatible UNet and LoRA components required by the protection script.
- `AntiMotion/assets/results/`: reserved space for README demos, qualitative comparisons, and generated previews.

## Method

AntiMotion uses a two-stage optimization process:

```text
Input video
  -> frame loading and resizing
  -> Stage 1: per-chunk PGD against Temporal LoRA adaptation
  -> Stage 2: cross-chunk PGD for temporal boundary disruption
  -> protected video
```

The objective combines a temporal denoising loss, an adversarial temporal difference branch, optional MetaCloak-style transform sampling, and a cross-chunk boundary term. The implementation is intentionally compact so experiments can be reproduced and modified directly from `AntiMotion/anti_motion.py`.

## Results and Video Demo

This section is reserved for qualitative results. Follow the MotionDirector-style README pattern: keep small GIF previews in `assets/` and show them with an HTML table. GIF previews render reliably on GitHub; MP4 files can be kept beside them as higher-quality sources.

### Demo Slots

| Case | Original input | Protected input | MotionDirector on original | MotionDirector on protected |
| --- | --- | --- | --- | --- |
| `case_01` | `AntiMotion/assets/results/case_01/original.gif` | `AntiMotion/assets/results/case_01/protected.gif` | `AntiMotion/assets/results/case_01/md_original.gif` | `AntiMotion/assets/results/case_01/md_protected.gif` |

After adding the GIF files above, replace the placeholder row with this block:

```html
<table>
  <tr>
    <td align="center"><b>Original input</b></td>
    <td align="center"><b>Protected input</b></td>
    <td align="center"><b>MotionDirector on original</b></td>
    <td align="center"><b>MotionDirector on protected</b></td>
  </tr>
  <tr>
    <td><img src="AntiMotion/assets/results/case_01/original.gif" width="220"></td>
    <td><img src="AntiMotion/assets/results/case_01/protected.gif" width="220"></td>
    <td><img src="AntiMotion/assets/results/case_01/md_original.gif" width="220"></td>
    <td><img src="AntiMotion/assets/results/case_01/md_protected.gif" width="220"></td>
  </tr>
</table>
```

Recommended demo file layout:

```text
AntiMotion/
`-- assets/
    `-- results/
        `-- case_01/
            |-- original.mp4
            |-- protected.mp4
            |-- md_original.mp4
            |-- md_protected.mp4
            |-- original.gif
            |-- protected.gif
            |-- md_original.gif
            `-- md_protected.gif
```

Recommended preview format:

- Use `.gif` for README inline previews.
- Keep each GIF at 320-480 px width and 8-12 fps when possible.
- Keep each README GIF small enough for GitHub to load quickly, preferably under 10 MB.
- Keep optional `.mp4` sources encoded as H.264/AVC with `yuv420p` pixel format.

Example conversion with `ffmpeg`:

```bash
ffmpeg -i AntiMotion/assets/results/case_01/original.mp4 \
  -vf "fps=8,scale=360:-1:flags=lanczos" \
  -loop 0 AntiMotion/assets/results/case_01/original.gif
```

Run the same conversion for `protected.mp4`, `md_original.mp4`, and `md_protected.mp4`.

## Project Structure

```text
anti_motion/
|-- README.md
`-- AntiMotion/
    |-- .gitignore
    |-- anti_motion.py
    |-- model_loader.py
    |-- requirements.txt
    |-- LICENSE
    |-- assets/
    |   `-- results/
    |       `-- case_01/
    |           `-- .gitkeep
    `-- third_party/
        `-- motiondirector/
            |-- models/
            |   |-- unet_3d_blocks.py
            |   `-- unet_3d_condition.py
            `-- utils/
                |-- convert_diffusers_to_original_ms_text_to_video.py
                |-- lora.py
                `-- lora_handler.py
```

## Installation

Create a clean Python environment. Python 3.10 or 3.11 is recommended.

```bash
cd AntiMotion

conda create -n antimotion python=3.11 -y
conda activate antimotion

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

For CUDA environments, install the PyTorch build that matches your driver and CUDA runtime first, then install the remaining requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Model Weights

Download a text-to-video base model separately. Model weights are intentionally not tracked in this repository.

```bash
hf download cerspense/zeroscope_v2_576w \
  --local-dir ./zeroscope_v2_576w
```

ModelScope T2V can also be used if the folder contains the expected Diffusers subfolders:

```bash
hf download ali-vilab/text-to-video-ms-1.7b \
  --local-dir ./modelscope_t2v_1.7b
```

Expected model layout:

```text
model_path/
|-- scheduler/
|-- tokenizer/
|-- text_encoder/
|-- vae/
`-- unet/
```

## Usage

The examples below use Linux shell syntax. On Windows PowerShell, set GPUs first with `$env:CUDA_VISIBLE_DEVICES="0,1"` and then run the `python anti_motion.py ...` command without the `CUDA_VISIBLE_DEVICES=0,1` prefix.

### Single Video

```bash
CUDA_VISIBLE_DEVICES=0,1 python anti_motion.py \
  --video_path /path/to/input.mp4 \
  --output_path outputs/protected_input.mp4 \
  --model_path ./zeroscope_v2_576w \
  --prompt "a person is moving" \
  --epsilon 0.05 \
  --pgd_steps 20 \
  --chunk_size 5 \
  --inner_steps 5 \
  --chaos_weight 0.2 \
  --global_weight 0.5
```

### Folder of Videos

```bash
CUDA_VISIBLE_DEVICES=0,1 python anti_motion.py \
  --video_path /path/to/video_folder \
  --output_path outputs/protected_videos \
  --model_path ./zeroscope_v2_576w \
  --prompt "a person is moving"
```

When `--video_path` is a single file, `--output_path` is treated as an output file. When `--video_path` is a folder, `--output_path` is treated as an output folder and each result is saved as `protected_<input_name>.mp4`.

## Main Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--video_path` | required | Input `.mp4` file or folder containing `.mp4` files. |
| `--output_path` | `protected.mp4` | Output file for a single input or output folder for batch mode. |
| `--model_path` | required | Local path to the base text-to-video model. |
| `--prompt` | `a person is moving` | Text prompt used by the protection objective. |
| `--width` | `576` | Width used when loading and resizing frames. |
| `--height` | `320` | Height used when loading and resizing frames. |
| `--epsilon` | `0.05` | Perturbation budget in `[0, 1]` pixel space. |
| `--pgd_steps` | `100` | Number of per-chunk PGD steps. |
| `--step_size` | auto | PGD step size. Defaults to `2.5 * epsilon / pgd_steps` after scaling. |
| `--chunk_size` | `5` | Number of frames per temporal chunk. |
| `--inner_steps` | `1` | Inner Temporal LoRA update steps. |
| `--inner_lr` | `1e-4` | Learning rate for inner LoRA updates. |
| `--surrogate_interval` | `10` | Steps between surrogate LoRA updates. |
| `--surrogate_lr` | `1e-5` | Learning rate for surrogate updates. |
| `--lora_rank` | `16` | Temporal LoRA rank. |
| `--chaos_weight` | `0.15` | Weight for intra-chunk temporal discontinuity. |
| `--global_weight` | `0.1` | Weight for cross-chunk boundary disruption. |
| `--global_pgd_steps` | `20` | Additional cross-chunk PGD steps. Set to `0` to disable Stage 2. |
| `--transform_sample_num` | `1` | Number of transform samples when transform sampling is enabled. |
| `--no_transform` | disabled | Disable transform sampling. |

## Hardware Notes

CUDA is strongly recommended. If multiple GPUs are visible, the script places the VAE on `cuda:1` and the UNet/text encoder on the main CUDA device to reduce memory pressure. CPU execution is available through PyTorch fallback but is not practical for normal experiments.

## Outputs

For each processed video, AntiMotion writes:

- A protected `.mp4` video.
- A terminal perturbation report containing epsilon, average L2 per frame, and PSNR.

Generated videos, base model weights, checkpoints, and experiment logs should stay outside git unless they are small README demo assets.

## Acknowledgements

This repository includes minimal MotionDirector-compatible components under `AntiMotion/third_party/motiondirector/`. Please follow the upstream licenses for MotionDirector, Diffusers, Transformers, ModelScope, and any base model weights used in your experiments.

## License

This project is released under the Apache-2.0 license. See [LICENSE](AntiMotion/LICENSE) for details.
