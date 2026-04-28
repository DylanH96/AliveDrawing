import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import torch

# The difference between img2img and just the raw controlnet is here we can change the strength,
# for rapid generations of just whatever crazy AI image Stable Diffusion comes up with just use
# controlnet, but for more tailoring but slower generation, use this interpolator

# loads original image and grayscale image
def load_img2img_pipeline():
    # same ControlNet model as controlnet_pipeline.py
    # still using canny edge maps as structural guide
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )

    # img2img variant accepts both original image and edge map
    # strength parameter controls how far AI transforms from original
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipeline.to("cuda")
    return pipeline

# takes both the original sketch and edge map as inputs unlike controlnet_pipeline.py
# strength parameter acts as a dial between 0.0 (original sketch) and 1.0 (full AI generation)
# allows smooth blending between the raw sketch and the concert visual output
def interpolate(pipeline, original, edge_map, prompt, strength=0.5, image_size=512):
    # resize original to match edge map dimensions
    # make sure to match image_size with preprocessor target_size when implementing in main
    original = cv.resize(original, (image_size, image_size), interpolation=cv.INTER_AREA)
    # original is BGR from OpenCV — convert to RGB before wrapping in PIL
    original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
    original = Image.fromarray(original)

    # edge map is single channel float32 0.0-1.0 from Stage 2
    # multiply back to uint8 and convert to RGB for diffusers
    edge_map = (edge_map * 255).astype(np.uint8)
    edge_map = Image.fromarray(edge_map).convert("RGB")

    negative_prompt = "blurry, low quality, dark, muddy colors, realistic photo"

    output = pipeline(
        prompt=prompt,
        image=original,
        control_image=edge_map,
        negative_prompt=negative_prompt,
        strength=strength,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    return output.images[0]

# this is generated and saved in one function
def generate_interpolated(original, edge_map, prompt, strength, image_size=512):
    # check for empty prompt param
    if prompt is None:
        raise ValueError("prompt required")

        # load img2img pipeline onto GPU
    pipeline = load_img2img_pipeline()

    # run interpolation with strength dial
    image = interpolate(pipeline, original, edge_map, prompt, strength, image_size)

    # save and return
    image.save(r"C:\AliveDrawing\image_saves\result.png")
    return image