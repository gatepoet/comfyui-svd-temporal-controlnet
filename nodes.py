import os
import torch
from torchvision.transforms import ToTensor

from .pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from .models.controlnet_sdv import ControlNetSDVModel
from .models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel

import comfy.model_management

script_directory = os.path.dirname(os.path.abspath(__file__))

class SVDTemporalControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "init_image": ("IMAGE", ),
            "control_frames": ("IMAGE", ),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "min_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "max_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
            "motion_bucket_id": ("INT", {"default": 100, "min": 1, "max": 4096, "step": 1}),
            "fps_id": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1}),
            "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 10.0, "step": 0.01}),
            "controlnet_cond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "checkpoint": (
            [   
                'stable-video-diffusion-img2vid',
                'stable-video-diffusion-img2vid-xt',
            ], {
               "default": 'DDIMScheduler'
            }),
            "controlnet": (
            [   
                'temporal-controlnet-lineart-svd-v1',
                'temporal-controlnet-depth-svd-v1',
            ], {
               "default": 'DDIMScheduler'
            }),
            },
            }
    
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES =("image","last_image")
    FUNCTION = "process"

    CATEGORY = "SVDTemporalControlnet"

    def process(self, init_image, control_frames, width, height, seed, steps, min_guidance_scale, max_guidance_scale, motion_bucket_id, fps_id, noise_aug_strength, controlnet_cond_scale, checkpoint, controlnet):
        
        comfy.model_management.unload_all_models()

        torch.manual_seed(seed)
        num_frames = control_frames.shape[0]
        init_image = init_image.permute(0, 3, 1, 2)  # Rearrange the tensor from [B, H, W, C] to [B, C, H, W]
        control_frames = control_frames.permute(0, 3, 1, 2)

        if not hasattr(self, 'pipeline'):
            # Load SVD checkpoint
            checkpoint_path = os.path.join(script_directory, f"../../models/diffusers/{checkpoint}")

            if os.path.exists(checkpoint_path):
                checkpoint_path = checkpoint_path
            else:
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=f"stabilityai/{checkpoint}", allow_patterns=["*.json","*fp16*"], local_dir=checkpoint_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError("No SVD model found.")
                
            # Load ControlNet checkpoint
            controlnet_path = os.path.join(script_directory, f"../../models/diffusers/{controlnet}")   
            if os.path.exists(controlnet_path):
                controlnet_path = controlnet_path
            else:
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=f"CiaraRowles/{controlnet}", local_dir=controlnet_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError("No controlnet model found.")
            
            # Load and set up the pipeline
            controlnet = ControlNetSDVModel.from_pretrained(controlnet_path,subfolder="controlnet", torch_dtype=torch.float16)
            unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(checkpoint_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
            pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(checkpoint_path,controlnet=controlnet,unet=unet, torch_dtype=torch.float16, variant="fp16")
            pipeline.enable_model_cpu_offload()

        # Inference and saving loop
        video_frames = pipeline(
            init_image, 
            control_frames, 
            width=width, 
            height=height, 
            decode_chunk_size=1,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps_id,
            noise_aug_strength = noise_aug_strength,
            controlnet_cond_scale=controlnet_cond_scale, 
            num_inference_steps=steps,
            min_guidance_scale = min_guidance_scale,
            max_guidance_scale = max_guidance_scale
            ).frames
        # Create an instance of the ToTensor transform
        to_tensor = ToTensor()

        # Apply the transform to each image in the list and stack them
        image_tensors = torch.stack([to_tensor(image) for image in [item for sublist in video_frames for item in sublist]])

        # Permute the tensor to the desired shape [B, H, W, C]
        image_tensors = image_tensors.permute(0, 2, 3, 1)
        
        return (image_tensors, image_tensors[-1].unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "SVDTemporalControlnet": SVDTemporalControlnet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDTemporalControlnet": "SVDTemporalControlnet",
}