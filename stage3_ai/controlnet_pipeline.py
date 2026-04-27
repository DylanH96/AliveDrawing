# for load_models()
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
# for run_pipeline()
from PIL import Image
import numpy as np

# [Claude - sonnet 4.6] notes:
# ControlNet — a neural network that takes a control image (our edge map) as structural guidance
# it constrains the AI to follow the shapes and lines of your sketch
# without ControlNet, Stable Diffusion would ignore your drawing entirely

# Stable Diffusion — a generative AI model that creates images from text prompts
# it works by starting with random noise and gradually denoising it into a coherent image
# the prompt drives the visual style — colors, lighting, mood, aesthetic

# together — ControlNet feeds your edge map as a blueprint
# Stable Diffusion paints a fully rendered image on top of that blueprint
# using whatever style you describe in the prompt
# your sketch's structure is preserved, the AI fills in the concert visuals

# downloads and caches ControlNet and Stable Diffusion model weights from Hugging Face
# float16 cuts VRAM usage in half so the models fit on the RTX 3060
# .to("cuda") moves all weights onto the GPU so inference runs locally at full speed

def load_models():
    # load ControlNet trained specifically on canny edge maps
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )

    # load Stable Diffusion with ControlNet attached
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet, # need controlnet for SD to read blueprint image (the edge map)
        torch_dtype=torch.float16
    )

    # move everything to GPU
    pipeline.to("cuda")
    return pipeline

# convert back to RGB values for PIL (Python Imaging Library) and outputs an AI generated image from the prompt
# PIl (through its modern fork Pillow) - a free, open-source Python library designed for opening,
# manipulating, and saving many different image file formats
def run_pipeline(pipeline, edge_map=None, prompt=None):
    # check for empty params
    if edge_map is None:
        raise ValueError("edge_map is required — pass the output of run_preprocessor()")
    if prompt is None:
        raise ValueError("prompt required my guy")

    # diffusers expects a PIL Image not a numpy array
    # convert float32 0.0-1.0 back to uint8 0-255 first
    edge_map = (edge_map * 255).astype(np.uint8)
    image = Image.fromarray(edge_map)

    # negative prompt steers the AI away from unwanted aesthetics
    negative_prompt = "blurry, low quality, dark, muddy colors, realistic photo"

    output = pipeline(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    )

    # output.images is a list — grab the first result
    return output.images[0]

# saves output bruh, returns PIL image
def save_output(image, output_path):
    # image is a PIL Image from the pipeline output
    # .save() infers file format from the extension in output_path
    image.save(output_path)
    return output_path

# avengers assemble! and return img, this file is the image generator
# need params here for placeholders for main (image and prompt), pipeline is always the same though
def generate(edge_map=None, prompt=None):
    pipeline = load_models()
    img = run_pipeline(pipeline, edge_map, prompt)
    save_output(img, r"C:\AliveDrawing\image_saves\result.png")
    return img