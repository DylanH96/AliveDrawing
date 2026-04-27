import cv2 as cv

# takes edge map array as param
def normalize(image):
    # don't need a for loop, can just divide the RGB values by 255 to get range from 0 to 1.0
    return image.astype("float32") / 255.0 # converts image array values to float
    # ControlNet and StableDiffusion are neural networks trained on float values 0.0 to 1.0
    # That's why we need to convert, keeps internal math stable and no huge numbers
    # same way you would convert to float 0-1.0 range for any machine learning context -
    # the model was trained expecting a certain value range and breaks if you give it something different

def resize_for_controlnet(image, target_size=512):
    # ControlNet was trained on 512x512 square images
    # forcing square dimensions here ensures Stage 3 AI gets expected input
    # unlike resize_to_max which preserved aspect ratio, this stretches to fit
    # INTER_AREA averages pixel neighborhoods for cleanest downsample quality
    resized = cv.resize(image, (target_size, target_size), interpolation=cv.INTER_AREA)
    return resized

def run_preprocessor(image):
    # resize before normalizing — always
    # normalizing first would change pixel values that resize math depends on
    image = resize_for_controlnet(image)
    image = normalize(image)
    return image

