import torch
import os
import clip
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from diffusers import StableDiffusionPipelineWithCLIP
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import Image


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

    def encode_image(self, image):
        with torch.no_grad():
            image = self.preprocess(image)
            vectors = [self.model_16.encode_image(image), self.model_32.encode_image(image), self.model_101.encode_image(image)]
            return torch.cat(vectors, dim=-1)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.model_16.apply(_convert_weights_to_fp16)
    model.model_32.apply(_convert_weights_to_fp16)
    model.model_101.apply(_convert_weights_to_fp16)


clip_model = MultiCLIP()
clip_model.to("cuda")
convert_weights(clip_model)

model_path = "/path_to_finetuned_stable_diffusion/finetuned_coco"
pipe = StableDiffusionPipelineWithCLIP.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

test_imgs = [fname for root, _dirs, files in os.walk("./test_imgs") for fname in files]
for img in test_imgs:
    input_img = ToTensor()(Image.open(os.path.join("./test_imgs", img))).unsqueeze(0).to("cuda")
    neg_img = torch.zeros(input_img.shape).to("cuda")
    image = pipe(prompt=None, prompt_embeds=clip_model.encode_image(image=input_img), negative_prompt_embeds=clip_model.encode_image(image=neg_img), img_emb=True).images[0]
    # image = pipe(prompt="a red bus on the street", img_emb=False).images[0] #
    image.save(os.path.join("./test_imgs", 're_' +img))