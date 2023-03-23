from pathlib import Path
from time import time
import datetime
import random
from tqdm import tqdm, trange
import numpy as np
import os
import sys
import copy
import torch
import clip

from model_lib.decoder.clip_prior import ClipPrior, Vocab
from model_lib.diffusion.script_util import create_sft_gaussian_diffusion as create_gaussian_diffusion_p2
from diffusers import StableDiffusionPipelineWithCLIP, EulerDiscreteScheduler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from transformers import AutoTokenizer, T5EncoderModel, T5Config


class MultiCLIP(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        model_32, _ = clip.load("./ViT-B-32.pt", device=device)
        model_16, _ = clip.load("./ViT-B-16.pt", device=device)
        model_101, _ = clip.load("./RN101.pt", device=device)
        self.model_32 = model_32
        self.model_16 = model_16
        self.model_101 = model_101
        self.preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def encode_image(self, image, dtype):
        with torch.no_grad():
            image = self.preprocess(image)
            vectors = [self.model_16.encode_image(image.to(dtype)), self.model_32.encode_image(image.to(dtype)), self.model_101.encode_image(image.to(dtype))]
            return torch.cat(vectors, dim=-1).to(dtype)

    def encode_text(self, text, dtype, device):
        with torch.no_grad():
            text = clip.tokenize(text).to(device)
            vectors = [self.model_16.encode_text(text), self.model_32.encode_text(text), self.model_101.encode_text(text)]
            return torch.cat(vectors, dim=-1).to(dtype)

def convert_weights(model: torch.nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, torch.nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def gen_clip_prior(model, diffusion, clip_emb, t5_emb, device="cuda"):
    B, C = clip_emb.shape[:2]
    uncond_clip_emb = clip_emb#torch.zeros_like(clip_emb)
    uncond_t5_emb = torch.zeros_like(t5_emb)
    model_kwargs = dict(
        clip_sentence_emb = torch.cat((clip_emb, uncond_clip_emb), dim=0),
        t5_word_emb = torch.cat((t5_emb, uncond_t5_emb), dim=0),
        emb_4_vocab=torch.cat((clip_emb, clip_emb), dim=0)
    )

    def cfg_sampling(x_t, ts, guidance_scale=1.0, **kwargs):
        # for sampling
        half = x_t[: len(x_t) // 2] # x_t: torch.Size([bx2, 3, 64, 64])
        combined = torch.cat([half, half], dim=0) # combined: torch.Size([bx2, 3, 64, 64])
        model_out = model(combined, ts, **kwargs)  # model_out: torch.Size([bx2, 6, 64, 64])
        eps, rest = torch.split(model_out, model_out.shape[1] //2, dim=1) # mean & variance
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)  # eps torch.Size([bx2, 3, 64, 64])
        return torch.cat([eps, rest], dim=1)  # torch.Size([bx2, 6, 64, 64])

    gen_im_emb = diffusion.p_sample_loop(cfg_sampling, (B *2, C), device=device, clip_denoised=False,
                                         progress=True, model_kwargs=model_kwargs, cond_fn=None)[ :B ]
    return gen_im_emb



prior_path = '/path_to_prior_model/model_state.th'
model_path = "/path_to_finetuned_stable_diffusion/finetuned_coco"
std_path = '/path_to_prior_model/std.pth'
mean_path = '/path_to_prior_model/mean.pth'
t5_device = "cpu"  #use cpu to load model if your GPU memory is limited
t5_model = 't5-11b'
scale = 5  # you can try different scales
layers = 16  # 16 for smaller model, 20 for larger model
log_std_init = torch.log(torch.load(std_path, map_location='cpu').view((-1, 1536)))[:1024].cuda()
mean_init = torch.load(mean_path, map_location='cpu').view((-1, 1536))[:1024].cuda()

with torch.no_grad():
    text = 'a yellow and blue train riding a track by some trees'
    clip_model = MultiCLIP()
    convert_weights(clip_model)
    t5_encoder = T5EncoderModel.from_pretrained(t5_model, low_cpu_mem_usage=True,
                                                torch_dtype=torch.float16).to(t5_device) # use float16 if you want to save time and memory
    tokenizer = AutoTokenizer.from_pretrained(t5_model, model_max_length=80)
    clip_model.cuda()
    # t5_encoder.to(dtype=torch.float16)
    t5_encoder.cuda()
    model = ClipPrior(xf_width=2048, xf_layers=layers, xf_heads=32,
                      clip_width=512*3, learn_sigma=False, use_vocab=True, vocab_size=1024,
                      vocab_use_mean=True, vocab_sample_num=1, t5_dim=t5_encoder.config.d_model,
                      vocab_log_std_init=log_std_init, vocab_mean_init=mean_init, vocab_learnable=False,
                      vocab_std_scale=scale, vocab_exp=False)
    ckpt = torch.load(prior_path)
    model.load_state_dict(ckpt)
    model.cuda()
    diffusion_fast = create_gaussian_diffusion_p2(
        steps=1000,
        learn_sigma=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=True,
        predict_prev=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="8",
        p2_gamma=1,
        p2_k=1,
        vocab=model.vocab,
        beta_min=0.0001,
        beta_max=0.02,
    )

    t5_ids = tokenizer(
        text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(t5_encoder.device)
    clip_text_emb = clip_model.encode_text(text, dtype=torch.float16, device="cuda")
    input_t5_emb = t5_encoder(input_ids=t5_ids).last_hidden_state.cuda()
    gen_emb = gen_clip_prior(model, diffusion_fast, clip_text_emb.float(), input_t5_emb.float())

    # NOTE: if you have a "ground-truth" image, and would like to compute cosine similarity
    # between generated embedding (gen_emb) and ground-truth embedding (gt_emb)
    # DO NOT use cosine_similarity(gen_emb, gt_emb, dim=-1) directly.
    # Our embeddings are concatenations of 3 CLIP embeddings,
    # you should use cosine_similarity(gen_emb.view(-1, 3, 512), gt_emb.view(-1, 3, 512), dim=-1)
    # to obtain similarities corresponding to different CLIP models
    # Also, you can also find that
    # cosine_similarity(gen_emb, gt_emb, dim=-1) != cosine_similarity(gen_emb.view(-1, 3, 512), gt_emb.view(-1, 3, 512), dim=-1).mean()
    # i.e. cosine_similarity(gen_emb, gt_emb, dim=-1) may give you misleading information
    # Also, similarity from different pre-trained CLIP of same image-text pair could be very different,
    # so DO NOT compare similarity across different pre-trained CLIP models.

    # use float32 if you have enough memory

    model.to(t5_device)  #move model to save memory if you have limited GPU memory

    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipelineWithCLIP.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.to("cuda")

    # classifier-free guidance
    # you can use a negative prompt or negative prompt embedding
    # depends on how you fine-tuning you stable diffusion

    # use negative prompt embedding
    # use which one as negative prompt embedding? depends on how you fine-tuning you stable diffusion

    negative_prompt = None

    # # 1. empty text generated embedding
    neg_prompt_embeds = torch.load("./small_sft_empty_embeds.th").cuda().to(dtype=torch.float16)

    # # this is generated by
    # # empty_t5_ids = tokenizer(
    # #     "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
    # #     return_tensors="pt"
    # # ).input_ids.to(t5_encoder.device)
    # # empty_t5_emb = t5_encoder(input_ids=empty_t5_ids).last_hidden_state.cuda()
    # # empty_emb = clip_model.encode_text("", dtype=torch.float16, device="cuda")
    # # empty_embeds = gen_clip_prior(model, diffusion_fast, empty_emb.float(), empty_t5_emb.float())
    #

    # # 2. ground-truth image embedding of "empty" image
    # neg_img = torch.zeros((1, 3, 224, 224)).to("cuda") #+ 0.5
    # neg_prompt_embeds = clip_model.encode_image(image=neg_img, dtype=torch.float16)
    #


    # # 3. use mean embeddings
    # neg_prompt_embeds = mean_init[:1].cuda()


    # # 4. use negative prompt
    # negative_prompt = ""
    # neg_prompt_embeds = None


    image = pipe(prompt=None, guidance_scale=2.0, num_inference_steps=100,
                 prompt_embeds=gen_emb.to(dtype=torch.float16), negative_prompt=negative_prompt,
                 negative_prompt_embeds=neg_prompt_embeds, img_emb=True).images[0]
    image.save(os.path.join("./test_imgs", f"prior_{text}.jpg"))