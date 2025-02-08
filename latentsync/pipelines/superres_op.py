import os
import torch
import cv2 as cv
from gfpgan.utils import GFPGANer
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img

if not os.path.exists(r'checkpoints/GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
    
def apply_codeformer(image, device, scale, target_size: tuple):
    ckpt_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    ckpt_path = load_file_from_url(url=ckpt_url, model_dir="weights/CodeFormer", progress=True)
    
    model = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
    ).to(device)
    
    checkpoint = torch.load(ckpt_path)["params_ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    
    img_tensor = img2tensor(image, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    img_tensor = (img_tensor - 0.5) /0.5
    
    with torch.no_grad():
        out = model(img_tensor, w = scale, adain = True)[0]
    final_img = tensor2img(out, rgb2bgr = True, min_max = (-1, 1))
    
    return cv.resize(final_img, target_size, cv.INTER_LANCZOS4)    
    
def apply_superres(image, scale, device, method: str, target_size: tuple):
    
    if method.lower() == "gfpgan":
        face_model = GFPGANer(model_path = r'checkpoints/GFPGANv1.4.pth', upscale = scale, arch = 'clean')
        try:
            _, _, img = face_model.enhance(image, paste_back = True)
        except RuntimeError as error:
            print('Error', error)
        
        img = cv.resize(img, target_size, interpolation = cv.INTER_LANCZOS4)
        # print(img.shape)        
        return img
        
    if method.lower() == "codeformer":
        img = apply_codeformer(image, device, scale, target_size)
        # print(img.shape)
        return img
        
        
    else:
        print(f"Unknown superresolution method: {method}. Skipping... use gfpgan or codeformer")
        return image