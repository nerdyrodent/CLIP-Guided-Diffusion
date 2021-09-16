# CLIP-Guided-Diffusion
Just playing with getting CLIP Guided Diffusion running locally, rather than having to use colab. 

Original by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings).
It uses a 512x512 unconditional ImageNet diffusion model fine-tuned from
OpenAI's 512x512 class-conditional ImageNet diffusion model (https://github.com/openai/guided-diffusion) together with 
CLIP (https://github.com/openai/CLIP) to connect text prompts with images.

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```
Guided Diffusion - https://github.com/openai/guided-diffusion
Katherine Crowson - <https://github.com/crowsonkb>
