# Install dependencies

git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/guided-diffusion
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips

# Download the diffusion models

curl -OL --http1.1 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
