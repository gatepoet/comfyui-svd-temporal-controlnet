# ComfyUI SVD Temporal Controlnet
https://github.com/kijai/comfyui-svd-temporal-controlnet/assets/40791699/1cc74616-939a-42a5-87d7-3fbca082fc35

ComfyUI wrapper node for Stable Video Diffusion Temporal Controlnet:
https://github.com/CiaraStrawberry/sdv_controlnet/

Work in progress, uses diffusers, thus hopefully a temporary solution until we have proper ComfyUI implementation.

Some requirements might be missing, let me know if so. These models (unnecessary files are filtered) should be automatically downloaded via hugginface_hub, if for some reason this fails, then they need to be placed into models/diffusers (if you're unfamiliar with diffusers, this means the whole folder, but from the model files only the fp16 versions are used currently):

https://huggingface.co/stabilityai/stable-video-diffusion-img2vid

https://huggingface.co/CiaraRowles/temporal-controlnet-depth-svd-v1

