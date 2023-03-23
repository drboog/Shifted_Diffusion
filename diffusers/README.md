<p align="center">
    <br>
    <img src="./docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

🤗 Diffusers provides pretrained diffusion models across multiple modalities, such as vision and audio, and serves
as a modular toolbox for inference and training of diffusion models.

More precisely, 🤗 Diffusers offers:

- State-of-the-art diffusion pipelines that can be run in inference with just a couple of lines of code (see [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)). Check [this overview](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/README.md#pipelines-summary) to see all supported pipelines and their corresponding official papers.
- Various noise schedulers that can be used interchangeably for the preferred speed vs. quality trade-off in inference (see [src/diffusers/schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)).
- Multiple types of models, such as UNet, can be used as building blocks in an end-to-end diffusion system (see [src/diffusers/models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)).
- Training examples to show how to train the most popular diffusion model tasks (see [examples](https://github.com/huggingface/diffusers/tree/main/examples), *e.g.* [unconditional-image-generation](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)).

## Installation

### For PyTorch

**With `pip`** (official package)
    
```bash
pip install --upgrade diffusers[torch]
```

**With `conda`** (maintained by the community)

```sh
conda install -c conda-forge diffusers
```

### For Flax

**With `pip`**

```bash
pip install --upgrade diffusers[flax]
```

**Apple Silicon (M1/M2) support**

Please, refer to [the documentation](https://huggingface.co/docs/diffusers/optimization/mps).

## Contributing

We ❤️  contributions from the open-source community! 
If you want to contribute to this library, please check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
You can look out for [issues](https://github.com/huggingface/diffusers/issues) you'd like to tackle to contribute to the library.
- See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities to contribute
- See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute exciting new diffusion models / diffusion pipelines
- See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Also, say 👋 in our public Discord channel <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>. We discuss the hottest trends about diffusion models, help each other with contributions, personal projects or
just hang out ☕.

## Quickstart

In order to get started, we recommend taking a look at two notebooks:

- The [Getting started with Diffusers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) notebook, which showcases an end-to-end example of usage for diffusion models, schedulers and pipelines.
  Take a look at this notebook to learn how to use the pipeline abstraction, which takes care of everything (model, scheduler, noise handling) for you, and also to understand each independent building block in the library.
- The [Training a diffusers model](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) notebook summarizes diffusion models training methods. This notebook takes a step-by-step approach to training your
  diffusion models on an image dataset, with explanatory graphics. 
  
## Stable Diffusion is fully compatible with `diffusers`!  

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/) and [RunwayML](https://runwayml.com/). It's trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 4GB VRAM.
See the [model card](https://huggingface.co/CompVis/stable-diffusion) for more information.


### Text-to-Image generation with Stable Diffusion

First let's install

```bash
pip install --upgrade diffusers transformers accelerate
```

We recommend using the model in [half-precision (`fp16`)](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) as it gives almost always the same results as full
precision while being roughly twice as fast and requiring half the amount of GPU RAM.

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
```

#### Running the model locally

You can also simply download the model folder and pass the path to the local folder to the `StableDiffusionPipeline`.

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

Assuming the folder is stored locally under `./stable-diffusion-v1-5`, you can run stable diffusion
as follows:

```python
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
```

If you are limited by GPU memory, you might want to consider chunking the attention computation in addition 
to using `fp16`.
The following snippet should result in less than 4GB VRAM.

```python
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()
image = pipe(prompt).images[0]  
```

If you wish to use a different scheduler (e.g.: DDIM, LMS, PNDM/PLMS), you can instantiate
it before the pipeline and pass it to `from_pretrained`.
    
```python
from diffusers import LMSDiscreteScheduler

pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```

If you want to run Stable Diffusion on CPU or you want to have maximum precision on GPU, 
please run the model in the default *full-precision* setting:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# disable the following line if you run on CPU
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```

### JAX/Flax

Diffusers offers a JAX / Flax implementation of Stable Diffusion for very fast inference. JAX shines specially on TPU hardware because each TPU server has 8 accelerators working in parallel, but it runs great on GPUs too.

Running the pipeline with the default PNDMScheduler:

```python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="flax", dtype=jax.numpy.bfloat16
)

prompt = "a photo of an astronaut riding a horse on mars"

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

**Note**:
If you are limited by TPU memory, please make sure to load the `FlaxStableDiffusionPipeline` in `bfloat16` precision instead of the default `float32` precision as done above. You can do so by telling diffusers to load the weights from "bf16" branch.

```python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jax.numpy.bfloat16
)

prompt = "a photo of an astronaut riding a horse on mars"

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

Diffusers also has a Image-to-Image generation pipeline with Flax/Jax
```python
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline

def create_key(seed=0):
    return jax.random.PRNGKey(seed)
rng = create_key(0)

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))

prompts = "A fantasy landscape, trending on artstation"

pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="flax",
    dtype=jnp.bfloat16,
)

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(prompt=[prompts]*num_samples, image = [init_img]*num_samples)
p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

output = pipeline(
    prompt_ids=prompt_ids, 
    image=processed_image, 
    params=p_params, 
    prng_seed=rng, 
    strength=0.75, 
    num_inference_steps=50, 
    jit=True, 
    height=512,
    width=768).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
```

Diffusers also has a Text-guided inpainting pipeline with Flax/Jax

```python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import PIL
import requests
from io import BytesIO


from diffusers import FlaxStableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained("xvjiarui/stable-diffusion-2-inpainting")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
init_image = num_samples * [init_image]
mask_image = num_samples * [mask_image]
prompt_ids, processed_masked_images, processed_masks = pipeline.prepare_inputs(prompt, init_image, mask_image)


# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)
processed_masked_images = shard(processed_masked_images)
processed_masks = shard(processed_masks)

images = pipeline(prompt_ids, processed_masks, processed_masked_images, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
```

### Image-to-Image text-guided generation with Stable Diffusion

The `StableDiffusionImg2ImgPipeline` lets you pass a text prompt and an initial image to condition the generation of new images.

```python
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

# or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# and pass `model_id_or_path="./stable-diffusion-v1-5"`.
pipe = pipe.to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb)

### In-painting using Stable Diffusion

The `StableDiffusionInpaintPipeline` lets you edit specific parts of an image by providing a mask and a text prompt.

```python
import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```

### Tweak prompts reusing seeds and latents

You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked.
Please have a look at [Reusing seeds for deterministic generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/reusing_seeds).

## Fine-Tuning Stable Diffusion

Fine-tuning techniques make it possible to adapt Stable Diffusion to your own dataset, or add new subjects to it. These are some of the techniques supported in `diffusers`:

Textual Inversion is a technique for capturing novel concepts from a small number of example images in a way that can later be used to control text-to-image pipelines. It does so by learning new 'words' in the embedding space of the pipeline's text encoder. These special words can then be used within text prompts to achieve very fine-grained control of the resulting images. 

- Textual Inversion. Capture novel concepts from a small set of sample images, and associate them with new "words" in the embedding space of the text encoder. Please, refer to [our training examples](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) or [documentation](https://huggingface.co/docs/diffusers/training/text_inversion) to try for yourself.

- Dreambooth. Another technique to capture new concepts in Stable Diffusion. This method fine-tunes the UNet (and, optionally, also the text encoder) of the pipeline to achieve impressive results. Please, refer to [our training example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) and [training report](https://huggingface.co/blog/dreambooth) for additional details and training recommendations.

- Full Stable Diffusion fine-tuning. If you have a more sizable dataset with a specific look or style, you can fine-tune Stable Diffusion so that it outputs images following those examples. This was the approach taken to create [a Pokémon Stable Diffusion model](https://huggingface.co/justinpinkney/pokemon-stable-diffusion) (by Justing Pinkney / Lambda Labs), [a Japanese specific version of Stable Diffusion](https://huggingface.co/spaces/rinna/japanese-stable-diffusion) (by [Rinna Co.](https://github.com/rinnakk/japanese-stable-diffusion/) and others. You can start at [our text-to-image fine-tuning example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and go from there.


## Stable Diffusion Community Pipelines

The release of Stable Diffusion as an open source model has fostered a lot of interesting ideas and experimentation. 
Our [Community Examples folder](https://github.com/huggingface/diffusers/tree/main/examples/community) contains many ideas worth exploring, like interpolating to create animated videos, using CLIP Guidance for additional prompt fidelity, term weighting, and much more! [Take a look](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview) and [contribute your own](https://huggingface.co/docs/diffusers/using-diffusers/contribute_pipeline).

## Other Examples

There are many ways to try running Diffusers! Here we outline code-focused tools (primarily using `DiffusionPipeline`s and Google Colab) and interactive web-tools.

### Running Code

If you want to run the code yourself 💻, you can try out:
- [Text-to-Image Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256)
```python
# !pip install diffusers["torch"] transformers
from diffusers import DiffusionPipeline

device = "cuda"
model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)
ldm = ldm.to(device)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
image = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images[0]

# save image
image.save("squirrel.png")
```
- [Unconditional Diffusion with discrete scheduler](https://huggingface.co/google/ddpm-celebahq-256)
```python
# !pip install diffusers["torch"]
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-celebahq-256"
device = "cuda"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to(device)

# run pipeline in inference (sample random noise and denoise)
image = ddpm().images[0]

# save image
image.save("ddpm_generated_image.png")
```
- [Unconditional Latent Diffusion](https://huggingface.co/CompVis/ldm-celebahq-256)
- [Unconditional Diffusion with continuous scheduler](https://huggingface.co/google/ncsnpp-ffhq-1024)

**Other Image Notebooks**:
* [image-to-image generation with Stable Diffusion](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb) ![Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/image_2_image_using_diffusers.ipynb),
* [tweak images via repeated Stable Diffusion seeds](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb) ![Open In Colab](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb),

**Diffusers for Other Modalities**:
* [Molecule conformation generation](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/geodiff_molecule_conformation.ipynb) ![Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/geodiff_molecule_conformation.ipynb),
* [Model-based reinforcement learning](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb) ![Open In Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb),

### Web Demos
If you just want to play around with some web demos, you can try out the following 🚀 Spaces:
| Model                          	| Hugging Face Spaces                                                                                                                                               	|
|--------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| Text-to-Image Latent Diffusion 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/text2img-latent-diffusion) 	|
| Faces generator                	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CompVis/celeba-latent-diffusion)    	|
| DDPM with different schedulers 	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/fusing/celeba-diffusion)           	|
| Conditional generation from sketch  	| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/huggingface/diffuse-the-rest)           	|
| Composable diffusion | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shuang59/Composable-Diffusion)           	|

## Definitions

**Models**: Neural network that models $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ (see image below) and is trained end-to-end to *denoise* a noisy input to an image.
*Examples*: UNet, Conditioned UNet, 3D UNet, Transformer UNet

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>
    
**Schedulers**: Algorithm class for both **inference** and **training**.
The class provides functionality to compute previous image according to alpha, beta schedule as well as predict noise for training. Also known as **Samplers**.
*Examples*: [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), [PNDM](https://arxiv.org/abs/2202.09778), [DEIS](https://arxiv.org/abs/2204.13902)

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349706-53d58acc-a4d1-4cda-b3e8-432d9dc7ad38.png" width="800"/>
    <br>
    <em> Sampling and training algorithms. Figure from DDPM paper (https://arxiv.org/abs/2006.11239). </em>
<p>
    

**Diffusion Pipeline**: End-to-end pipeline that includes multiple diffusion models, possible text encoders, ...
*Examples*: Glide, Latent-Diffusion, Imagen, DALL-E 2

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174348898-481bd7c2-5457-4830-89bc-f0907756f64c.jpeg" width="550"/>
    <br>
    <em> Figure from ImageGen (https://imagen.research.google/). </em>
<p>
    
## Philosophy

- Readability and clarity is preferred over highly optimized code. A strong importance is put on providing readable, intuitive and elementary code design. *E.g.*, the provided [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) are separated from the provided [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) and provide well-commented code that can be read alongside the original paper.
- Diffusers is **modality independent** and focuses on providing pretrained models and tools to build systems that generate **continuous outputs**, *e.g.* vision and audio.
- Diffusion models and schedulers are provided as concise, elementary building blocks. In contrast, diffusion pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box, should stay as close as possible to their original implementation and can include components of another library, such as text-encoders. Examples for diffusion pipelines are [Glide](https://github.com/openai/glide-text2im) and [Latent Diffusion](https://github.com/CompVis/latent-diffusion).

## In the works

For the first release, 🤗 Diffusers focuses on text-to-image diffusion techniques. However, diffusers can be used for much more than that! Over the upcoming releases, we'll be focusing on:

- Diffusers for audio
- Diffusers for reinforcement learning (initial work happening in https://github.com/huggingface/diffusers/pull/105).
- Diffusers for video generation
- Diffusers for molecule generation (initial work happening in https://github.com/huggingface/diffusers/pull/54)

A few pipeline components are already being worked on, namely:

- BDDMPipeline for spectrogram-to-sound vocoding
- GLIDEPipeline to support OpenAI's GLIDE model
- Grad-TTS for text to audio generation / conditional audio generation

We want diffusers to be a toolbox useful for diffusers models in general; if you find yourself limited in any way by the current API, or would like to see additional models, schedulers, or techniques, please open a [GitHub issue](https://github.com/huggingface/diffusers/issues) mentioning what you would like to see.

## Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:

- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim).
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
