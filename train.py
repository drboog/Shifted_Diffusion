from pathlib import Path
import logging
from time import time
import datetime
import random
from tqdm import tqdm, trange
import numpy as np
import os
import sys
import copy
import math
import torch
import argparse
import transformers
import clip
from packaging import version
import datasets
from datasets import load_dataset

from transformers import AutoTokenizer, T5EncoderModel, T5Config, T5ForConditionalGeneration
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from model_lib.decoder.clip_prior import ClipPrior, Vocab
from model_lib.diffusion.script_util import create_sft_gaussian_diffusion as create_gaussian_diffusion_p2
from model_lib.diffusion.resample import create_named_schedule_sampler as create_named_schedule_sampler_p2
# from utils.checkpoint import save_checkpoint, load_from_pretrain

logger = get_logger(__name__, log_level="INFO")

class MultiCLIP(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        model_32, _ = clip.load("./ViT-B-32.pt", device=device)
        model_16, _ = clip.load("./ViT-B-16.pt", device=device)
        model_101, _ = clip.load("./RN101.pt", device=device)
        self.model_32 = model_32
        self.model_16 = model_16
        self.model_101 = model_101
        # self.preprocess = Compose([
        #     Resize(224, interpolation=BICUBIC),
        #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])

    def encode_image(self, image, dtype):
        with torch.no_grad():
            # image = self.preprocess(image)
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

@torch.no_grad()
def get_t5_embeddings(texts, t5_encoder, t5_tokenizer):
    input_ids = t5_tokenizer( texts , return_tensors="pt", padding="max_length", truncation=True, max_length=80).input_ids.to(t5_encoder.device)
    outputs = t5_encoder( input_ids=input_ids ).last_hidden_state
    return outputs # in shape: B x Seq_len x Hidden_size (1024)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.96, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.06, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-06, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument('--initial_lg_loss_scale', type=float, default=10.0,
    #                     help="initial log loss scale for mix precision")
    parser.add_argument(
        "--noise_schedule",
        type=str,
        default="linear",
    )
    parser.add_argument("--t5_model", type=str, default="t5-11b", help="Path to T5 model.")
    parser.add_argument("--empty_t5_prob", type=float, default=0.1, help="Probability of dropping t5 text embeddings.")
    parser.add_argument("--empty_clip_prob", type=float, default=0., help="Probability of dropping clip embeddings.")
    parser.add_argument("--use_vocab", type=bool, default=True, help="Use vocab which contains mean and std for Gaussian clusters.")
    parser.add_argument("--vocab_learnable", type=bool, default=False, help="The vocab is learnable or not.")
    parser.add_argument("--exp", type=bool, default=False, help="Use exponential scaling or not.")
    parser.add_argument("--size", type=int, default=1024, help="Size of vocab.")
    parser.add_argument("--use_mean", type=bool, default=True, help="Use Gaussian cluster means or samples to assign input embedding to clusters.")
    parser.add_argument("--sample_num", type=int, default=1, help="If use samples from clusters, how many samples to be used.")
    parser.add_argument("--mean_path", type=str, help="Path to mean of Gaussians of your dataset. If you didn't analyze your dataset to obtain it manually, you can use the file in our pre-trained models.")
    parser.add_argument("--std_path", type=str, help="Path to std of Gaussians of your dataset. If you didn't  analyze your dataset to obtain it manually, you can use the file in our pre-trained models.")
    parser.add_argument("--beta_min", type=float, default=0.0001, help="Beta for diffusion.")
    parser.add_argument("--beta_max", type=float, default=0.02, help="Beta for diffusion.")
    parser.add_argument("--std_scale", type=float, default=5.0, help="Scaling factor for std.")
    parser.add_argument("--p2_gamma", type=float, default=1.0, help="P2 weighting")
    parser.add_argument("--vocab_lr_scale", type=float, default=0.01, help="Scaling learning rate for vocab when learnable.")
    parser.add_argument("--model_width", type=int, default=2048)
    parser.add_argument("--model_layers", type=int, default=8)
    parser.add_argument("--model_num_heads", type=int, default=32)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    logging_dir = os.path.join(args.output_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    log_std_init = torch.log(torch.load(args.std_path, map_location='cpu').view((-1, 1536)))[:args.size]
    mean_init = torch.load(args.mean_path, map_location='cpu').view((-1, 1536))[:args.size]
    # lg_loss_scale = args.initial_lg_loss_scale

    logger.info(" Loading pre-trained text encoders, may take some time if download is needed.")
    clip_model = MultiCLIP()
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        # convert_weights(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model, model_max_length=80)
    t5_encoder = T5EncoderModel.from_pretrained(args.t5_model, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    clip_model.requires_grad_(False)
    t5_encoder.requires_grad_(False)

    model = ClipPrior(xf_width=args.model_width, xf_layers=args.model_layers, xf_heads=args.model_num_heads,
                      clip_width=512*3, learn_sigma=False, t5_dim=t5_encoder.config.d_model, use_vocab=args.use_vocab,
                      vocab_size=args.size, vocab_use_mean=args.use_mean, vocab_sample_num=args.sample_num,
                      vocab_log_std_init=log_std_init, vocab_mean_init=mean_init, vocab_learnable=args.vocab_learnable,
                      vocab_std_scale=args.std_scale, vocab_lr_scale=args.vocab_lr_scale, vocab_exp=args.exp)

    # if args.gradient_checkpointing: # TODO
    #     model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay
    )

    data_files = {}
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
    )
    column_names = dataset["train"].column_names
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )

    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids, captions

    train_transforms = Compose(
        [ToTensor(),
         Resize(224, interpolation=BICUBIC),
         CenterCrop(224),
         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
         ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"], examples["text"]  = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        text = [example["text"] for example in examples]
        return {"pixel_values": pixel_values, "input_ids": input_ids, "text": text}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    clip_model.to(accelerator.device)
    t5_encoder.to(accelerator.device, dtype=weight_dtype)
    try:
        vocab = model.module.vocab
    except:
        vocab = model.vocab

    if not args.vocab_learnable:
        vocab.mean = mean_init.to(accelerator.device)
        vocab.std = log_std_init.exp().to(accelerator.device)

    diffusion = create_gaussian_diffusion_p2(steps=1000,
                                            learn_sigma=False,
                                            noise_schedule=args.noise_schedule,
                                            use_kl=False,
                                            predict_xstart=True,
                                            predict_prev=False,
                                            rescale_timesteps=False,
                                            rescale_learned_sigmas=False,
                                            timestep_respacing="",
                                            p2_gamma=args.p2_gamma,
                                            p2_k=1,
                                            vocab = vocab,
                                            beta_min=args.beta_min,
                                            beta_max=args.beta_max,
                                             )
    schedule_sampler = create_named_schedule_sampler_p2("uniform", diffusion)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("prior", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                clip_image_emb = clip_model.encode_image(batch["pixel_values"], dtype=weight_dtype)
                clip_text_emb = clip_model.encode_text(batch["text"], dtype=weight_dtype, device=accelerator.device)
                # input_t5_emb = get_t5_embeddings(text, t5_encoder, t5_tokenizer)
                input_t5_emb = t5_encoder(input_ids=batch["input_ids"]).last_hidden_state

                input_clip_emb_modified = clip_text_emb.detach().clone()
                clip_empty_idx = torch.rand(clip_image_emb.shape[0]) <= args.empty_clip_prob
                input_clip_emb_modified[clip_empty_idx] *= 0.0

                input_t5_emb_modified = input_t5_emb.detach().clone()
                t5_empty_idx = torch.rand(clip_image_emb.shape[0]) <= args.empty_t5_prob
                input_t5_emb_modified[t5_empty_idx] *= 0.0

                t, weights = schedule_sampler.sample(clip_image_emb.shape[0], accelerator.device, p=None, weights_np=None) # weights shape (batch_size,)
                losses = diffusion.training_losses(model, clip_image_emb, t, #emb_4_vocab=clip_image_emb,
                                                   model_kwargs=dict(emb_4_vocab=clip_text_emb, #clip_image_emb,
                                                    t5_word_emb=input_t5_emb_modified,
                                                    clip_sentence_emb=input_clip_emb_modified
                                                    ), use_d=False, discriminator=None)
                loss = (losses["loss"] * weights).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            if global_step >= args.max_train_steps:
                break

    # # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        accelerator.save(model.state_dict(), os.path.join(args.output_dir, 'prior.pt'))
    accelerator.end_training()

if __name__ == "__main__":
    main()
