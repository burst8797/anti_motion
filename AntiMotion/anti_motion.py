"""Generate adversarial video perturbations for MotionDirector Temporal LoRA."""

import argparse
import gc
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model_loader import load_primary_models
from third_party.motiondirector.utils.lora import extract_lora_child_module
from third_party.motiondirector.utils.lora_handler import LoraHandler


def load_video(video_path, width=576, height=320):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        raw.append(frame)
    cap.release()

    if not raw:
        raise ValueError(f"No frames from: {video_path}")

    frames = np.stack(raw).astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
    frames = frames * 2.0 - 1.0
    print(f"  Loaded {frames.shape[0]} frames [{width}x{height}] fps={fps:.1f}")
    return frames, fps


def save_video(frames, output_path, fps=8):
    frames = ((frames.clamp(-1, 1) + 1) / 2 * 255).byte()
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved to {output_path}")


def load_model(model_path, device):
    print(f"  Loading model: {model_path}")

    scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model_path)

    vae_device = "cuda:1" if torch.cuda.device_count() > 1 else device

    vae = vae.to(vae_device, dtype=torch.float32).eval()
    unet = unet.to(device, dtype=torch.float32).eval()
    text_encoder = text_encoder.to(device, dtype=torch.float32).eval()

    for m in [vae, text_encoder]:
        for p in m.parameters():
            p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    return vae, unet, text_encoder, tokenizer, scheduler, vae_device


@torch.no_grad()
def encode_prompt(prompt, tokenizer, text_encoder, device):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(tokens)[0]


def init_temporal_lora(unet, lora_rank=16, lora_dropout=0.1):
    lora_manager = LoraHandler(
        use_unet_lora=True, unet_replace_modules=["TransformerTemporalModel"]
    )
    lora_params, _ = lora_manager.add_lora_to_model(
        True, unet, lora_manager.unet_replace_modules, lora_dropout, "", r=lora_rank
    )
    return lora_params, lora_manager


def get_lora_params(unet):
    loras = extract_lora_child_module(
        unet, target_replace_module=["TransformerTemporalModel"]
    )
    params = []
    for lora in loras:
        for p in lora.parameters():
            params.append(p)
    return params


def compute_temporal_loss(
    unet,
    vae,
    scheduler,
    enc_hidden,
    chunk: torch.Tensor,
    device,
    vae_device,
    chaos_weight: float = 0.15,
) -> torch.Tensor:
    """
    L_temp  = MSE(pred, noise)
    L_ad    = MSE(alpha * pred - beta * pred[random_idx],
                  alpha * noise - beta * noise[random_idx])
    L_chaos = mean((pred[:, t + 1] - pred[:, t]) ** 2)
    """
    num_frames = chunk.shape[0]

    lats = vae.encode(chunk.to(vae_device)).latent_dist.sample()
    lats = lats * vae.config.scaling_factor
    lats_vid = lats.to(device).permute(1, 0, 2, 3).unsqueeze(0)

    noise = torch.randn_like(lats_vid)
    t = torch.randint(400, 600, (1,), device=device)
    noisy = scheduler.add_noise(lats_vid, noise, t)

    pred = unet(noisy, t, encoder_hidden_states=enc_hidden).sample

    loss_temp = F.mse_loss(pred, noise)

    beta = 1.0
    alpha = (beta**2 + 1) ** 0.5
    random_idx = torch.randint(0, num_frames, (1,), device=device).item()
    pred_ad = alpha * pred - beta * pred[:, :, random_idx, :, :].unsqueeze(2)
    target_ad = alpha * noise - beta * noise[:, :, random_idx, :, :].unsqueeze(2)
    loss_ad = F.mse_loss(pred_ad, target_ad)

    if pred.shape[2] >= 2:
        pred_diff = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        loss_chaos = pred_diff.pow(2).mean()
    else:
        loss_chaos = torch.tensor(0.0, device=device)

    return loss_temp + loss_ad + chaos_weight * loss_chaos


def random_transform(frames: torch.Tensor) -> torch.Tensor:
    transformed = TF.gaussian_blur(
        frames, kernel_size=5, sigma=random.uniform(0.1, 2.0)
    )
    if random.random() < 0.5:
        transformed = TF.hflip(transformed)
    return transformed


def compute_metacloak_transform_loss(
    unet,
    vae,
    scheduler,
    enc_hidden,
    frames: torch.Tensor,
    device,
    vae_device,
    chaos_weight: float = 0.15,
    sample_num: int = 1,
) -> torch.Tensor:
    sample_num = max(1, sample_num)
    loss = 0.0
    for _ in range(sample_num):
        transformed = random_transform(frames)
        loss = loss + compute_temporal_loss(
            unet,
            vae,
            scheduler,
            enc_hidden,
            transformed,
            device,
            vae_device,
            chaos_weight,
        )
    return loss / sample_num


def pgd_one_chunk(
    chunk,
    vae,
    unet,
    scheduler,
    enc_hidden,
    device,
    vae_device,
    epsilon,
    pgd_steps,
    step_size,
    inner_steps,
    inner_lr,
    use_transform,
    lora_params,
    surrogate_interval=10,
    surrogate_lr=1e-5,
    chaos_weight=0.15,
    transform_sample_num=1,
):
    chunk = chunk.to(device)

    delta = ((torch.rand_like(chunk) * 2 - 1) * epsilon * 0.1).detach()
    delta.requires_grad_(True)

    surrogate_optim = torch.optim.SGD(lora_params, lr=surrogate_lr)

    for step in range(pgd_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        lora_snapshot = [p.data.clone() for p in lora_params]

        for p in lora_params:
            p.requires_grad_(True)
        delta.requires_grad_(False)

        inner_optim = torch.optim.SGD(lora_params, lr=inner_lr)
        for _ in range(inner_steps):
            inner_optim.zero_grad()
            loss_inner = compute_temporal_loss(
                unet,
                vae,
                scheduler,
                enc_hidden,
                chunk.detach(),
                device,
                vae_device,
                chaos_weight,
            )
            loss_inner.backward()
            inner_optim.step()

        delta.requires_grad_(True)
        for p in lora_params:
            p.requires_grad_(False)

        adv_outer = (chunk + delta).clamp(-1, 1)

        if use_transform:
            loss_outer = compute_metacloak_transform_loss(
                unet,
                vae,
                scheduler,
                enc_hidden,
                adv_outer,
                device,
                vae_device,
                chaos_weight,
                sample_num=transform_sample_num,
            )
        else:
            loss_outer = compute_temporal_loss(
                unet,
                vae,
                scheduler,
                enc_hidden,
                adv_outer,
                device,
                vae_device,
                chaos_weight,
            )
        loss_outer.backward()

        with torch.no_grad():
            delta_new = delta + step_size * delta.grad.sign()
            delta_new = delta_new.clamp(-epsilon, epsilon)
            delta_new = (chunk + delta_new).clamp(-1, 1) - chunk
            delta.copy_(delta_new)

        for p, snap in zip(lora_params, lora_snapshot):
            p.data.copy_(snap)

        if (step + 1) % surrogate_interval == 0:
            adv_current = (chunk + delta.detach()).clamp(-1, 1)
            for p in lora_params:
                p.requires_grad_(True)
            surrogate_optim.zero_grad()
            loss_surr = compute_temporal_loss(
                unet,
                vae,
                scheduler,
                enc_hidden,
                adv_current.detach(),
                device,
                vae_device,
                chaos_weight,
            )
            loss_surr.backward()
            surrogate_optim.step()
            for p in lora_params:
                p.requires_grad_(False)

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"    step [{step + 1:3d}/{pgd_steps}]  "
                f"inner_loss={loss_inner.item():.4f}  "
                f"outer_loss={loss_outer.item():.4f}"
            )

    protected = (chunk + delta.detach()).clamp(-1, 1)
    return protected.cpu()


def find_max_chunk_pair(
    chunks_adv,
    vae,
    unet,
    scheduler,
    enc_hidden,
    device,
    vae_device,
):
    n = len(chunks_adv)
    preds = []

    with torch.no_grad():
        for chunk in chunks_adv:
            chunk = chunk.to(device)
            lats = vae.encode(chunk.to(vae_device)).latent_dist.sample()
            lats = lats * vae.config.scaling_factor
            lats_vid = lats.to(device).permute(1, 0, 2, 3).unsqueeze(0)
            noise = torch.randn_like(lats_vid)
            t = torch.randint(400, 600, (1,), device=device)
            noisy = scheduler.add_noise(lats_vid, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=enc_hidden).sample
            preds.append(pred)

    max_diff = -1.0
    best_i, best_j = 0, min(1, n - 1)

    for i in range(n):
        for j in range(i + 1, n):
            diff = (
                (preds[i][:, :, -1, :, :] - preds[j][:, :, 0, :, :])
                .pow(2)
                .mean()
                .item()
            )
            if diff > max_diff:
                max_diff = diff
                best_i, best_j = i, j

    return best_i, best_j


def pgd_multi_chunk(
    chunks,
    vae,
    unet,
    scheduler,
    enc_hidden,
    device,
    vae_device,
    epsilon,
    pgd_steps,
    step_size,
    inner_steps,
    inner_lr,
    use_transform,
    lora_params,
    surrogate_interval=10,
    surrogate_lr=1e-5,
    chaos_weight=0.15,
    global_weight=0.1,
    transform_sample_num=1,
):
    n = len(chunks)
    chunks = [c.to(device) for c in chunks]

    deltas = []
    for c in chunks:
        d = ((torch.rand_like(c) * 2 - 1) * epsilon * 0.05).detach()
        d.requires_grad_(True)
        deltas.append(d)

    surrogate_optim = torch.optim.SGD(lora_params, lr=surrogate_lr)

    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  [Stage 2] Cross-chunk PGD  chunks={n}  steps={pgd_steps}")
    print(f"  chaos_weight={chaos_weight}  global_weight={global_weight}")
    print(SEP)

    for step in range(pgd_steps):

        adv_chunks = [(c + d).clamp(-1, 1) for c, d in zip(chunks, deltas)]
        adv_t = [random_transform(a) if use_transform else a for a in adv_chunks]

        best_i, best_j = find_max_chunk_pair(
            adv_t, vae, unet, scheduler, enc_hidden, device, vae_device
        )

        lora_snapshot = [p.data.clone() for p in lora_params]

        for p in lora_params:
            p.requires_grad_(True)
        for d in deltas:
            d.requires_grad_(False)

        inner_optim = torch.optim.SGD(lora_params, lr=inner_lr)
        for _ in range(inner_steps):
            inner_optim.zero_grad()
            loss_inner = (
                compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    chunks[best_i].detach(),
                    device,
                    vae_device,
                    chaos_weight,
                )
                + compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    chunks[best_j].detach(),
                    device,
                    vae_device,
                    chaos_weight,
                )
            )
            loss_inner.backward()
            inner_optim.step()

        for p in lora_params:
            p.requires_grad_(False)

        for k, d in enumerate(deltas):
            d.requires_grad_(k in (best_i, best_j))
            if d.grad is not None:
                d.grad.zero_()

        base_adv_i = (chunks[best_i] + deltas[best_i]).clamp(-1, 1)
        base_adv_j = (chunks[best_j] + deltas[best_j]).clamp(-1, 1)

        sample_num = max(1, transform_sample_num) if use_transform else 1
        loss_outer = 0.0
        loss_global = 0.0
        for _ in range(sample_num):
            adv_i = random_transform(base_adv_i) if use_transform else base_adv_i
            adv_j = random_transform(base_adv_j) if use_transform else base_adv_j

            lats_i = (
                vae.encode(adv_i.to(vae_device)).latent_dist.sample()
                * vae.config.scaling_factor
            )
            lats_j = (
                vae.encode(adv_j.to(vae_device)).latent_dist.sample()
                * vae.config.scaling_factor
            )
            lats_i_v = lats_i.to(device).permute(1, 0, 2, 3).unsqueeze(0)
            lats_j_v = lats_j.to(device).permute(1, 0, 2, 3).unsqueeze(0)
            noise_i = torch.randn_like(lats_i_v)
            noise_j = torch.randn_like(lats_j_v)
            t = torch.randint(400, 600, (1,), device=device)
            pred_i = unet(
                scheduler.add_noise(lats_i_v, noise_i, t),
                t,
                encoder_hidden_states=enc_hidden,
            ).sample
            pred_j = unet(
                scheduler.add_noise(lats_j_v, noise_j, t),
                t,
                encoder_hidden_states=enc_hidden,
            ).sample

            sample_global = (pred_i[:, :, -1, :, :] - pred_j[:, :, 0, :, :]).pow(2).mean()
            loss_global = loss_global + sample_global
            loss_outer = loss_outer + (
                compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    adv_i,
                    device,
                    vae_device,
                    chaos_weight,
                )
                + compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    adv_j,
                    device,
                    vae_device,
                    chaos_weight,
                )
                + global_weight * sample_global
            )
        loss_outer = loss_outer / sample_num
        loss_global = loss_global / sample_num
        loss_outer.backward()

        with torch.no_grad():
            for idx in [best_i, best_j]:
                d_new = deltas[idx] + step_size * deltas[idx].grad.sign()
                d_new = d_new.clamp(-epsilon, epsilon)
                d_new = (chunks[idx] + d_new).clamp(-1, 1) - chunks[idx]
                deltas[idx].copy_(d_new)

        for p, snap in zip(lora_params, lora_snapshot):
            p.data.copy_(snap)

        if (step + 1) % surrogate_interval == 0:
            for p in lora_params:
                p.requires_grad_(True)
            surrogate_optim.zero_grad()
            loss_surr = (
                compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    (chunks[best_i] + deltas[best_i].detach()).clamp(-1, 1).detach(),
                    device,
                    vae_device,
                    chaos_weight,
                )
                + compute_temporal_loss(
                    unet,
                    vae,
                    scheduler,
                    enc_hidden,
                    (chunks[best_j] + deltas[best_j].detach()).clamp(-1, 1).detach(),
                    device,
                    vae_device,
                    chaos_weight,
                )
            )
            loss_surr.backward()
            surrogate_optim.step()
            for p in lora_params:
                p.requires_grad_(False)

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"    step [{step + 1:3d}/{pgd_steps}]  "
                f"pair=({best_i},{best_j})  "
                f"inner={loss_inner.item():.4f}  "
                f"outer={loss_outer.item():.4f}  "
                f"global={loss_global.item():.4f}"
            )

    protected = [(c + d.detach()).clamp(-1, 1).cpu() for c, d in zip(chunks, deltas)]
    return protected


def protect_video(
    frames,
    vae,
    unet,
    scheduler,
    enc_hidden,
    device,
    vae_device,
    epsilon=0.05,
    pgd_steps=100,
    step_size=None,
    chunk_size=5,
    inner_steps=1,
    inner_lr=1e-4,
    surrogate_interval=10,
    surrogate_lr=1e-5,
    use_transform=True,
    lora_rank=16,
    chaos_weight=0.15,
    global_weight=0.1,
    global_pgd_steps=20,
    transform_sample_num=1,
):
    if step_size is None:
        step_size = 2.5 * epsilon / pgd_steps

    F_total = frames.shape[0]
    if F_total < 2:
        raise ValueError("Anti-Motion temporal protection requires at least 2 frames.")

    chunks = [
        (s, min(s + chunk_size, F_total), frames[s:min(s + chunk_size, F_total)])
        for s in range(0, F_total, chunk_size)
    ]
    if len(chunks) > 1 and chunks[-1][2].shape[0] == 1:
        prev_start, _, _ = chunks[-2]
        _, last_end, _ = chunks[-1]
        chunks[-2] = (prev_start, last_end, frames[prev_start:last_end])
        chunks.pop()

    SEP = "-" * 60
    print(f"\n{SEP}")
    print(f"  Protection run  frames={F_total}  chunk={chunk_size}  chunks={len(chunks)}")
    print(f"  PGD: steps={pgd_steps}  epsilon={epsilon}  step={step_size:.5f}")
    print(f"  Inner update: steps={inner_steps}  lr={inner_lr}")
    print(f"  Surrogate update: every {surrogate_interval} steps  lr={surrogate_lr}")
    print(f"  chaos_weight={chaos_weight}  global_weight={global_weight}")
    print(
        "  Transform sampling: "
        f"{'MetaCloak branch' if use_transform else 'disabled'}  "
        f"samples={transform_sample_num if use_transform else 0}"
    )
    print(SEP)

    lora_params = get_lora_params(unet)
    if len(lora_params) == 0:
        print("  Initializing Temporal LoRA...")
        init_temporal_lora(unet, lora_rank=lora_rank)
        lora_params = get_lora_params(unet)
    print(f"  Temporal LoRA parameter tensors: {len(lora_params)}")

    print(f"\n  [Stage 1] Per-chunk PGD  steps={pgd_steps}...")
    protected_chunks = []
    for i, (start, end, chunk) in enumerate(chunks):
        print(f"\n  chunk [{i + 1}/{len(chunks)}]  frames {start}-{end - 1}")
        pc = pgd_one_chunk(
            chunk,
            vae,
            unet,
            scheduler,
            enc_hidden,
            device,
            vae_device,
            epsilon,
            pgd_steps,
            step_size,
            inner_steps,
            inner_lr,
            use_transform,
            lora_params,
            surrogate_interval=surrogate_interval,
            surrogate_lr=surrogate_lr,
            chaos_weight=chaos_weight,
            transform_sample_num=transform_sample_num,
        )
        protected_chunks.append(pc)
        torch.cuda.empty_cache()
        gc.collect()

    if len(protected_chunks) >= 2 and global_pgd_steps > 0:
        protected_chunks = pgd_multi_chunk(
            protected_chunks,
            vae,
            unet,
            scheduler,
            enc_hidden,
            device,
            vae_device,
            epsilon,
            global_pgd_steps,
            step_size,
            inner_steps,
            inner_lr,
            use_transform,
            lora_params,
            surrogate_interval=surrogate_interval,
            surrogate_lr=surrogate_lr,
            chaos_weight=chaos_weight,
            global_weight=global_weight,
            transform_sample_num=transform_sample_num,
        )
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(protected_chunks, dim=0)


def report(original, protected, epsilon):
    delta = protected - original
    l2 = delta.flatten(1).norm(dim=1).mean().item()
    mse = delta.pow(2).mean().item()
    psnr = 10 * math.log10(4.0 / mse) if mse > 0 else float("inf")
    SEP = "-" * 60
    print(f"\n{SEP}")
    print("  Perturbation report")
    print(SEP)
    print(f"  epsilon   : {epsilon:.4f}")
    print(f"  L2/frame  : {l2:.4f}")
    print(f"  PSNR      : {psnr:.1f} dB")
    print(SEP)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate video perturbations for MotionDirector Temporal LoRA."
    )
    p.add_argument("--video_path", required=True)
    p.add_argument("--output_path", default="protected.mp4")
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompt", default="a person is moving")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--pgd_steps", type=int, default=100)
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--chunk_size", type=int, default=5)
    p.add_argument(
        "--inner_steps", type=int, default=1, help="Inner LoRA update steps."
    )
    p.add_argument(
        "--inner_lr", type=float, default=1e-4, help="Inner update learning rate."
    )
    p.add_argument(
        "--surrogate_interval",
        type=int,
        default=10,
        help="Steps between surrogate updates.",
    )
    p.add_argument(
        "--surrogate_lr",
        type=float,
        default=1e-5,
        help="Surrogate update learning rate.",
    )
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument(
        "--chaos_weight",
        type=float,
        default=0.15,
        help="Weight for intra-chunk temporal discontinuity loss.",
    )
    p.add_argument(
        "--global_weight",
        type=float,
        default=0.1,
        help="Weight for cross-chunk boundary discontinuity loss.",
    )
    p.add_argument(
        "--global_pgd_steps",
        type=int,
        default=20,
        help="Cross-chunk PGD steps. Set to 0 to disable stage 2.",
    )
    p.add_argument(
        "--transform_sample_num",
        type=int,
        default=1,
        help="Number of MetaCloak transform samples.",
    )
    p.add_argument("--no_transform", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLI epsilon is defined in [0, 1] pixel space; tensors use [-1, 1].
    epsilon_scaled = args.epsilon * 2

    print(f"\n{'='*60}")
    print(f"  Anti-Motion  |  device={device}")
    print("  Method: two-stage PGD for MotionDirector Temporal LoRA")
    print(f"{'='*60}")

    print("\n[1/4] Loading model...")
    vae, unet, text_encoder, tokenizer, scheduler, vae_device = load_model(
        args.model_path, device
    )

    print(f"\n[2/4] Encoding prompt: '{args.prompt}'")
    with torch.no_grad():
        enc_hidden = encode_prompt(args.prompt, tokenizer, text_encoder, device)

    if os.path.isdir(args.video_path):
        video_files = sorted(
            [
                os.path.join(args.video_path, f)
                for f in os.listdir(args.video_path)
                if f.lower().endswith(".mp4")
            ]
        )
        os.makedirs(args.output_path, exist_ok=True)
    else:
        video_files = [args.video_path]

    print(f"\n[3/4] Found {len(video_files)} video(s)")

    for i, vf in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(video_files)}] {os.path.basename(vf)}")
        frames, fps = load_video(vf, args.width, args.height)

        print("\n[4/4] Running protection...")
        protected = protect_video(
            frames=frames,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            enc_hidden=enc_hidden,
            device=device,
            vae_device=vae_device,
            epsilon=epsilon_scaled,
            pgd_steps=args.pgd_steps,
            step_size=args.step_size,
            chunk_size=args.chunk_size,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            surrogate_interval=args.surrogate_interval,
            surrogate_lr=args.surrogate_lr,
            use_transform=not args.no_transform,
            lora_rank=args.lora_rank,
            chaos_weight=args.chaos_weight,
            global_weight=args.global_weight,
            global_pgd_steps=args.global_pgd_steps,
            transform_sample_num=args.transform_sample_num,
        )

        report(frames, protected, epsilon_scaled)

        out_file = (
            os.path.join(args.output_path, f"protected_{os.path.basename(vf)}")
            if os.path.isdir(args.video_path)
            else args.output_path
        )
        save_video(protected, out_file, fps)

        del frames, protected
        torch.cuda.empty_cache()
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
