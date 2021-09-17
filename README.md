# CLIP-Guided-Diffusion
Just playing with getting CLIP Guided Diffusion running locally, rather than having to use colab. 

Original colab notebooks by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings):

* Original 256x256 notebook: [![Open In Colab][colab-badge]][colab-notebook1]

[colab-notebook1]: <https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj#scrollTo=X5gODNAMEUCR>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

It uses OpenAI's 256x256 unconditional ImageNet diffusion model (https://github.com/openai/guided-diffusion)

* Original 512x512 notebook: [![Open In Colab][colab-badge]][colab-notebook2]

[colab-notebook2]: <https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA#scrollTo=VnQjGugaDZPJ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

It uses a 512x512 unconditional ImageNet diffusion model fine-tuned from OpenAI's 512x512 class-conditional ImageNet diffusion model (https://github.com/openai/guided-diffusion)

Together with CLIP (https://github.com/openai/CLIP), they connect text prompts with images.

Either the 256 or 512 model can be used here (by setting `--output_size` to either 256 or 512)

## Environment
* Ubuntu 20.04 (Windows untested but should work)
* Anaconda
* Nvidia RTX 3090

Typical VRAM requirments:
* 256 defaults: 10 GB
* 512 defaults: 18 GB

## Set up

This example uses [Anaconda](https://www.anaconda.com/products/individual#Downloads) to manage virtual Python environments.

Create a new virtual Python environment for CLIP-Guided-Diffusion:
```sh
conda create --name cgd python=3.9
conda activate cgd
```

Run the setup file:
```sh
./setup.sh
```

Or if you want to run the commands manually:
```sh
# Install dependencies

git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/guided-diffusion
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install lpips

# Download the diffusion models

curl -OL --http1.1 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
```
## Inference

To just use the defaults along with your text, you can run:

```sh
python generate_diffuse.py "A painting of an apple"
```

An example using a number of options:
```sh
python generate_diffuse.py -p "An amazing fractal" -os=256 -cgs=1000 -tvs=50 -rgs=50 -cuts=16 -cutb=4 -t=200 -se=200 -m ViT-B/32 -o my_fractal.png
```

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```
* Guided Diffusion - https://github.com/openai/guided-diffusion
* Katherine Crowson - <https://github.com/crowsonkb>
