from diffusers import DDIMScheduler

try:
    from diffusers.models import AutoencoderKL
except ImportError:
    from diffusers import AutoencoderKL

from third_party.motiondirector.models.unet_3d_condition import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_primary_models(pretrained_model_path):
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    return scheduler, tokenizer, text_encoder, vae, unet
