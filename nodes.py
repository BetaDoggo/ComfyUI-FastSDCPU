import json
import requests
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image

class fastsdcpu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,}),
                "negative_prompt": ("STRING", {"multiline": True,}),
                "width": (["256","512","768","1024",], {"default:": 2}),
                "height": (["256","512","768","1024",], {"default:": 2}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1, "min": 1, "max": 20, "step": 0.5,}),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 5}),
                "use_taesd": ("BOOLEAN", {"default": True},),
                "token_merging": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.1,}),
                "endpoint": ("STRING", {"default": "http://localhost:8000",}),
            },
            "optional": {
                "openvino_model": ("STRING",),
                "lcm_model": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "fastsdcpu"

    def generate(self, prompt, negative_prompt, width, height, steps, cfg, seed, clip_skip, use_taesd, token_merging, endpoint, openvino_model=None, lcm_model=None):
        body = {
            #"lcm_model_id": lcm_model,
            "openvino_lcm_model_id": openvino_model,
            #"use_offline_model": False,
            #"use_lcm_lora": False,
            #"lcm_lora": {
            #    "base_model_id": "Lykon/dreamshaper-8",
            #    "lcm_lora_id": "latent-consistency/lcm-lora-sdv1-5"
            #},
            "use_tiny_auto_encoder": use_taesd,
            "use_openvino": openvino_model != "" or None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            #"init_image": "string",
            #"strength": 0.6,
            "image_height": height,
            "image_width": width,
            "inference_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "token_merging": token_merging,
            "number_of_images": 1,
            "seed": seed,
            "use_seed": True,
            "diffusion_task": "text_to_image",
            "rebuild_pipeline": False
        }

        response = requests.post(endpoint + "/api/generate", data=json.dumps(body),)
        image = Image.open(BytesIO(base64.b64decode(response.json()['images'][0])))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

class fastsdcpu_vino_models:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["Disty0/LCM_SoteMix","rupeshs/sd-turbo-openvino","rupeshs/sdxs-512-0.9-openvino","rupeshs/hyper-sd-sdxl-1-step-openvino-int8","rupeshs/SDXL-Lightning-2steps-openvino-int8","rupeshs/sdxl-turbo-openvino-int8","rupeshs/LCM-dreamshaper-v7-openvino","rupeshs/FLUX.1-schnell-openvino-int4","rupeshs/sd15-lcm-square-openvino-int8",],),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "choose"
    CATEGORY = "fastsdcpu"

    def choose(self, model):
        return (model,)
    
class fastsdcpu_lcm_models:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["stabilityai/sd-turbo","rupeshs/sdxs-512-0.9-orig-vae","rupeshs/hyper-sd-sdxl-1-step","rupeshs/SDXL-Lightning-2steps","stabilityai/sdxl-turbo","SimianLuo/LCM_Dreamshaper_v7","latent-consistency/lcm-sdxl","latent-consistency/lcm-ssd-1b",],),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "choose"
    CATEGORY = "fastsdcpu"

    def choose(self, model):
        return (model,)

NODE_CLASS_MAPPINGS = {
    "fastsdcpu": fastsdcpu,
    "fastsdcpu_vino_models": fastsdcpu_vino_models,
    "fastsdcpu_lcm_models": fastsdcpu_lcm_models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "fastsdcpu": "fastsdcpu",
    "fastsdcpu_vino_models": "fastsdcpu_vino_models",
    "fastsdcpu_lcm_models": "fastsdcpu_lcm_models",
}