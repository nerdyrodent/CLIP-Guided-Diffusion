# Generates images from text prompts with CLIP guided diffusion.

# Original by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings).
# It uses a 512x512 unconditional ImageNet diffusion model fine-tuned from
# OpenAI's 512x512 class-conditional ImageNet diffusion model (https://github.com/openai/guided-diffusion) together with 
# CLIP (https://github.com/openai/CLIP) to connect text prompts with images. 
# Modifications by Nerdy Rodent (https://github.com/nerdyrodent, https://twitter.com/NerdyRodent).

# Licensed under the MIT License

# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# Imports
import argparse
import gc
import io
import math
import sys
import os

#from IPython import display
import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# Testing
#import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np

# Video stuff
from subprocess import Popen, PIPE, run
import re

# Args
# Create the parser
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments
vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default="A painting of a nerdy rodent", dest='prompts')
vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')

vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
vq_parser.add_argument("-st",   "--skip_steps", type=int, help="Skip steps for init image (200-500)", default=0, dest='skip_timesteps') # This needs to be between approx. 200 and 500 when using an init image.
vq_parser.add_argument("-is",   "--init_scale", type=int, help="Initial image scale (e.g. 1000)", default=0, dest='init_scale') # This enhances the effect of the init image, a good value is 1000.

vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/16', dest='clip_model')

vq_parser.add_argument("-t",    "--timesteps", type=str, help="Number of timesteps", default='1000', dest='timesteps') # number(s) (Can be comma separated) or one of ddim25, ddim50, ddim150, ddim250, ddim500, ddim1000 (must be mod0 of diffusion_steps)
vq_parser.add_argument("-ds",   "--diffusion_steps", type=int, help="Diffusion steps", default=1000, dest='diffusion_steps')
vq_parser.add_argument("-se",   "--save_every", type=int, help="Image update frequency", default=100, dest='save_every')

vq_parser.add_argument("-bs",   "--batch_size", type=int, help="Batch size", default=1, dest='batch_size')
vq_parser.add_argument("-nb",   "--num_batches", type=int, help="Number of batches", default=1, dest='n_batches')

vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=16, dest='cutn')
vq_parser.add_argument("-cutb", "--cutn_batches", type=int, help="Number of cut batches", default=2, dest='cutn_batches') # Gradient accumulate every
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')

vq_parser.add_argument("-cgs",  "--clip_scale", type=int, help="CLIP guidance scale", default=1000, dest='clip_guidance_scale') # Controls how much the image should look like the prompt.
vq_parser.add_argument("-tvs",  "--tv_scale", type=float, help="Smoothness scale", default=150, dest='tv_scale') # Controls the smoothness of the final output.
vq_parser.add_argument("-rgs",  "--range_scale", type=int, help="RGB range scale", default=50, dest='range_scale') # Controls how far out of range RGB values are allowed to be.

vq_parser.add_argument("-os",   "--output_size", type=int, help="Output image size (256 or 512)", default=256, dest='image_size')
vq_parser.add_argument("-s",    "--seed", type=int, help="Seed", default=None, dest='seed')
vq_parser.add_argument("-o",    "--output", type=str, help="Output file", default="output.png", dest='output')

vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames (steps)?", dest='make_video')
vq_parser.add_argument("-vup",  "--video_upscale", action='store_true', help="Upscale video? (needs Real-ESRGAN executable)", dest='upscale_video')

vq_parser.add_argument("-nfp",  "--no_fp16", action='store_false', help="Disable fp16?", dest='use_fp16')
vq_parser.add_argument("-nbm",  "--no_benchmark", action='store_false', help="Disable CuDNN benchmark?", dest='cudnn_bm')
vq_parser.add_argument("-pl",   "--plot_loss", action='store_true', help="Plot loss?", dest='graph_loss')

vq_parser.add_argument("-dev",  "--cuda_device", type=str, help="CUDA Device", default='cuda:0', dest='cuda_device')


# Execute the parse_args() method
args = vq_parser.parse_args()

if args.image_size != 256 and args.image_size != 512:
    args.image_size = 256

# Make video steps directory
if args.make_video:
    if not os.path.exists('steps'):
        os.mkdir('steps')

# Use all the things!
if args.cudnn_bm:
    torch.backends.cudnn.benchmark = True

# Settings
# Text prompts
prompts = [phrase.strip() for phrase in args.prompts.split("|")]

# Image prompts
if args.image_prompts:
    args.image_prompts = args.image_prompts.split("|")
    args.image_prompts = [image.strip() for image in args.image_prompts]    
    
image_prompts = args.image_prompts
batch_size = args.batch_size
clip_guidance_scale = args.clip_guidance_scale
tv_scale = args.tv_scale
range_scale = args.range_scale
cutn = args.cutn
cutn_batches = args.cutn_batches
cut_pow = args.cut_pow
n_batches = args.n_batches
init_image = args.init_image
skip_timesteps = args.skip_timesteps
init_scale = args.init_scale
seed = args.seed


# Define necessary functions
def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow      

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
            #cutouts.append(F.adaptive_max_pool2d(cutout, self.cut_size))

        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


# Model settings
model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': args.diffusion_steps,
    'rescale_timesteps': True,
    'timestep_respacing': args.timesteps,
    'image_size': args.image_size,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': args.use_fp16,
    'use_scale_shift_norm': True,
})

# Load models
device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Size: ', args.image_size)

model, diffusion = create_model_and_diffusion(**model_config)
if args.image_size == 256:
    model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
else:
    model.load_state_dict(torch.load('512x512_diffusion_uncond_finetune_008100.pt', map_location='cpu'))

model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

clip_model = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)


### Actually do the run...
def do_run():
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed  
    torch.manual_seed(seed)
    print("Seed:", seed)
    
    loss_test = []

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
    side_x = side_y = model_config['image_size']

    target_embeds, weights = [], []

    for prompt in prompts:
        print('Text prompt:', prompt)
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        print('Image prompt:', prompt)    
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / cutn] * cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        print('Initial image:', init_image)
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            
            # Encode image and calculate spherical distance loss
            for i in range(cutn_batches):
                clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
                image_embeds = clip_model.encode_image(clip_in).float()
                dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
                dists = dists.view([cutn, n, -1])
                losses = dists.mul(weights).sum(2).mean(0)

                # Saving loss for plot
                if args.graph_loss:
                    loss_test.append(losses.sum().item())                
                                
                x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
            
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])            
            loss = (tv_losses.sum() * tv_scale) + (range_losses.sum() * range_scale)
            
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
                
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            return grad

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):
        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
        )

        for j, sample in enumerate(samples):
            if j % args.save_every == 0 or cur_t == 0:
                print()
                for k, image in enumerate(sample['pred_xstart']):
                    #filename = f'progress_{i * batch_size + k:05}.png'
                    #TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    if i > 0:
                        b_filename = (os.path.basename(args.output).split('.')[0])
                        b_filename = b_filename + '-' + str(i) + '.png'
                        b_filename = os.path.join(os.path.dirname(args.output), b_filename)
                    else:
                        b_filename = args.output

                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(b_filename)
                    tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    #display.display(display.Image(filename))

                if args.graph_loss:
                    plt.plot(np.array(loss_test), 'r')    
                    plot_filename = (os.path.basename(args.output).split('.')[0])
                    plot_filename = plot_filename + '-' + str(i) + '-loss.jpg'
                    plot_filename = os.path.join(os.path.dirname(args.output), plot_filename)
                    plt.savefig(plot_filename)
            
            if args.make_video:
                for k, image in enumerate(sample['pred_xstart']):
                    step_filename = str(j) + '.png'
                    frame_filename = os.path.join('steps', step_filename)
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(frame_filename)            
            
            # Countdown
            cur_t -= 1

        # Video generation
        if args.make_video:
            init_frame = 1      # Initial video frame
            last_frame = j      # This will raise an error if that number of frames does not exist.

            length = 5          # Desired time of the video in seconds

            min_fps = 10
            max_fps = 60
            
            total_frames = last_frame-init_frame
            
            if args.upscale_video:
                if not os.path.exists('upscaled_steps'):
                    os.mkdir('upscaled_steps')
                try:
                    run("./realesrgan-ncnn-vulkan -i steps -o upscaled_steps", check=True, shell=True)
                except FileNotFoundError:
                    print("realesrgan-ncnn-vulkan not found")
                    args.upscale_video = False

            frames = []
            tqdm.write('Generating video...')
            for k in range(init_frame,last_frame):
                if args.upscale_video:
                    temp = Image.open("./upscaled_steps/"+ str(k) +'.png')
                else:
                    temp = Image.open("./steps/"+ str(k) +'.png')
                keep = temp.copy()
                frames.append(keep)
                temp.close()
            
            fps = np.clip(total_frames/length,min_fps,max_fps)

            # Batches
            if i > 0:
                m_filename = (os.path.basename(args.output).split('.')[0])
                m_filename = m_filename + '-' + str(i) + '.png'
                m_filename = os.path.join(os.path.dirname(args.output), m_filename)
            else:
                m_filename = args.output

            output_file = re.compile('\.png$').sub('.mp4', m_filename)
            
            try:
                p = Popen(['ffmpeg',
                           '-y',
                           '-f', 'image2pipe',
                           '-vcodec', 'png',
                           '-r', str(fps),
                           '-i',
                           '-',
                           '-vcodec', 'libx264',
                           '-r', str(fps),
                           '-pix_fmt', 'yuv420p',
                           '-crf', '17',
                           '-preset', 'veryslow',
                           output_file], stdin=PIPE)
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")        
            for im in tqdm(frames):
                im.save(p.stdin, 'PNG')
            p.stdin.close()
            p.wait()

# Run   
if __name__ == "__main__":
    gc.collect()
    do_run()         
