import os, shutil, requests, random, time, uuid, boto3, runpod
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime

import torch
import numpy as np
from PIL import Image

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

import comfy
from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import nodes_model_advanced

from comfy_execution.utils import get_executing_context
if get_executing_context() is None:
    from types import SimpleNamespace
    import comfy_execution.utils
    comfy_execution.utils.get_executing_context = lambda: SimpleNamespace(node_id="notebook_manual_run")

import asyncio
asyncio.run(load_custom_node("/content/ComfyUI/custom_nodes/seedvr2_videoupscaler", module_parent="custom_nodes"))

LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
SeedVR2LoadDiTModel = NODE_CLASS_MAPPINGS["SeedVR2LoadDiTModel"]()
SeedVR2LoadVAEModel = NODE_CLASS_MAPPINGS["SeedVR2LoadVAEModel"]()
SeedVR2VideoUpscaler = NODE_CLASS_MAPPINGS["SeedVR2VideoUpscaler"]()

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
ModelSamplingAuraFlow = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()

with torch.inference_mode():
    seedvr2_dit = SeedVR2LoadDiTModel.execute(model="seedvr2_ema_7b_fp16.safetensors", device="cuda:0", offload_device="cpu", cache_model=False, blocks_to_swap=35, swap_io_components=False,
                                        attention_mode="sdpa")[0]
    seedvr2_vae = SeedVR2LoadVAEModel.execute(model="ema_vae_fp16.safetensors", device="cuda:0", offload_device="cpu", cache_model= False, encode_tiled=True,encode_tile_size=1024,
                                      encode_tile_overlap=128,decode_tiled=True, decode_tile_size=1024,decode_tile_overlap=128,tile_debug= "false")[0]
    unet = UNETLoader.load_unet("z_image_turbo_bf16.safetensors", "default")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]

@torch.inference_mode()
def generate(input):
    try:
        tmp_dir="/content/ComfyUI/output"
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:6]
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        s3_access_key_id = os.getenv('s3_access_key_id')
        s3_secret_access_key = os.getenv('s3_secret_access_key')
        s3_endpoint_url = os.getenv('s3_endpoint_url')
        s3_region_name = os.getenv('s3_region_name')
        s3_bucket_name = os.getenv('s3_bucket_name')
        s3_bucket_folder = os.getenv('s3_bucket_folder')
        s3 = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, endpoint_url=s3_endpoint_url, region_name=s3_region_name)

        values = input["input"]
        job_id = values['job_id']

        resolution = values['resolution'] # 4000

        positive_prompt = values['positive_prompt']
        negative_prompt = values['negative_prompt']
        seed = values['seed'] # 0
        steps = values['steps'] # 9
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # euler
        scheduler = values['scheduler'] # simple
        denoise = values['denoise'] # 1.0
        width = values['width'] # 1024
        height = values['height'] # 1024
        batch_size = values['batch_size'] # 1.0
        input_lora = values.get('input_lora')
        shift = values['shift'] # 3.0

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 2**32 - 1)
        
        if input_lora:
            input_lora_strength = values['input_lora_strength'] # 1.0
            input_lora = download_file(url=input_lora, save_dir="/content/ComfyUI/models/loras", file_name='input_lora')
            input_lora = os.path.basename(input_lora)
            unet_lora = LoraLoaderModelOnly.load_lora_model_only(unet, input_lora, strength_model=input_lora_strength)[0]

        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
        latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
        
        model = unet_lora if input_lora else unet
        model_patch = ModelSamplingAuraFlow.patch_aura(model, shift)[0]

        samples = KSampler.sample(model_patch, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        comfy.model_management.unload_all_models()
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"{tmp_dir}/z_image_turbo.png")
        comfy.model_management.unload_all_models()

        input_image = LoadImage.load_image(f"{tmp_dir}/z_image_turbo.png")[0]
        upscaled_image = SeedVR2VideoUpscaler.execute(image=input_image, dit=seedvr2_dit, vae=seedvr2_vae, seed=seed, resolution=resolution, max_resolution=resolution, batch_size=1,uniform_batch_size=False, temporal_overlap=16,
                                     prepend_frames=0, color_correction="lab", input_noise_scale=0.0, latent_noise_scale=0.0, offload_device="cpu", enable_debug=False)[0]
        Image.fromarray(np.array(upscaled_image*255, dtype=np.uint8)[0]).save(f"{tmp_dir}/z_image_turbo_seedvr2.png")
        comfy.model_management.unload_all_models()

        result = f"{tmp_dir}/z_image_turbo_seedvr2.png"
        
        s3_key =  f"{s3_bucket_folder}/z_image_turbo_seedvr2-{current_time}-{seed}-{unique_id}.png"
        s3.upload_file(result, s3_bucket_name, s3_key, ExtraArgs={'ContentType': 'image/png'})
        result_url = f"{s3_endpoint_url}/{s3_bucket_name}/{s3_key}"

        return {"job_id": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        return {"job_id": job_id, "result": str(e), "status": "FAILED"}
    finally:
        directory_path = Path(tmp_dir)
        if directory_path.exists():
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been removed successfully.")
        else:
            print(f"Directory {directory_path} does not exist.")

runpod.serverless.start({"handler": generate})