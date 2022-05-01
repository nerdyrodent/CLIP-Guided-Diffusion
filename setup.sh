# Install dependencies
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/guided-diffusion
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips matplotlib IPython requests

# Download the diffusion models
if ! [ -f ./512x512_diffusion_uncond_finetune_008100.pt ]; then
    curl -OL --http1.1 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'  
fi

if ! [ -f ./256x256_diffusion_uncond.pt ]; then
    curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
fi
