import os
import json
import requests
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import folder_paths

class fastsdcpu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,}),
                "negative_prompt": ("STRING", {"multiline": True,}),
                "width": (["256","512","768","1024",],),
                "height": (["256","512","768","1024",],),
                "steps": ("INT", {"default": 1, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1, "min": 1, "max": 20, "step": 0.5,}),
                "seed": ("INT", {"default": 1337, "min": 1, "max": 16777215}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 16}),               
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 5}),
                "token_merging": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.1,}),
                "use_taesd": ("BOOLEAN", {"default": True},),
                "use_seed": ("BOOLEAN", {"default": False},),
                "use_local_path": ("BOOLEAN", {"default": False}),
                #"apply_LCM_lora": ("BOOLEAN", {"default": False}),
                "endpoint": ("STRING", {"default": "http://localhost:8000",}),
            },
            "optional": {
                "openvino_model": ("STRING",),
                "lcm_model": ("STRING",),
                "i2i_strength": ("FLOAT", {"default": 0.75, "min": 0, "max": 1, "step": 0.01,}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("Image","Latency",)
    FUNCTION = "generate"
    CATEGORY = "fastsdcpu"

    def generate(self, prompt, negative_prompt, width, height, steps, cfg, seed, batch_size, batch_count, clip_skip, token_merging, use_taesd, use_seed, use_local_path, endpoint, openvino_model="", lcm_model="", i2i_strength=None, image=None):
        #main args
        body = {
            "use_offline_model": use_local_path,
            #"use_lcm_lora": apply_LCM_lora,
            "openvino_lcm_model_id": "filler", #an input is required even if it's not used (I think)
            "use_tiny_auto_encoder": use_taesd,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_height": height,
            "image_width": width,
            "inference_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "token_merging": token_merging,
            "number_of_images": batch_size,
            "use_seed": use_seed,
            "diffusion_task": "text_to_image",
            "rebuild_pipeline": False
        }
        #enable i2i
        if image is not None:
            image_np = 255. * image.cpu().numpy().squeeze()
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(image_np)
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_params = {
                "init_image": img_b64,
                "strength": i2i_strength,
                "diffusion_task": "image_to_image",
            }
            body.update(image_params)
        #select model type
        if openvino_model != "":
            models = {
                "use_openvino": True,
                "openvino_lcm_model_id": openvino_model,
            }
        else:
            models = {
                "use_openvino": False,
                "lcm_model_id": lcm_model,
            }      
        body.update(models)
        '''if apply_LCM_lora:
            lcm_lora = {
                "lcm_lora": {
                    "base_model_id": openvino_model if openvino_model is not None and not "" else lcm_model,
                    "lcm_lora_id": "latent-consistency/lcm-lora-sdv1-5"
                },
            }
            body.update(lcm_lora)'''
        image_list = []
        for batch in range(batch_count):
            seeds = {
                "seed": seed + batch - 1,
            }
            body.update(seeds)
            print("Request: \n" +  str(body))
            response = requests.post(endpoint + "/api/generate", data=json.dumps(body),)
            for output in range(batch_size):
                image = Image.open(BytesIO(base64.b64decode(response.json()['images'][output - 1])))
                image = np.array(image).astype(np.float32) / 255.0
                image_list.append(torch.from_numpy(image))
        images = torch.stack(image_list, dim=0)
        return (images,str(response.json()['latency']) + " Seconds",)

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
    
class fastsdcpu_loadModel:
    @classmethod
    def INPUT_TYPES(cls):
        model_path = os.path.join(folder_paths.models_dir, "diffusers")
        models = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        return {
            "required": {
                "model": (models,),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "choose"
    CATEGORY = "fastsdcpu"

    def choose(self, model):
        return (os.path.join(folder_paths.models_dir, "diffusers", model),)

NODE_CLASS_MAPPINGS = {
    "fastsdcpu": fastsdcpu,
    "fastsdcpu_vino_models": fastsdcpu_vino_models,
    "fastsdcpu_lcm_models": fastsdcpu_lcm_models,
    "fastsdcpu_loadModel": fastsdcpu_loadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "fastsdcpu": "fastsdcpu",
    "fastsdcpu_vino_models": "fastsdcpu_vino_models",
    "fastsdcpu_lcm_models": "fastsdcpu_lcm_models",
    "fastsdcpu_loadModel": "fastsdcpu_loadModel",
}