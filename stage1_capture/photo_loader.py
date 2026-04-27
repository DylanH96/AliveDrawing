# let's try pathlib instead of os, this accesses the files on our computer
import pathlib as pl
# OpenCV handles anything that is direct image manipulation — reading files, resizing,
# converting color spaces, thresholding, denoising, edge detection.
# It's fast, runs on CPU, has no setup headache, and is deterministic (same input always gives same output).
import cv2 as cv
# This file is meant for loading, giving data on image, and resizing

# reads image
def load_image(image_path, color_mode):
    if validate_path(image_path):
        img = cv.imread(image_path)

        if color_mode.lower() == "grayscale":
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif color_mode.lower() == "rgb":
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported color mode: {color_mode}")

        return img
    else:
        raise FileNotFoundError("path does not exist")

# checks if image path is valid
def validate_path(image_path):
    # want to validate all requirements for it to return true
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"] # .webp is for testing image for now,
    # ideally not included .webp
    if not pl.Path(image_path).is_file():
        raise ValueError(f"Just not a file bro: {pl.Path(image_path)}")
    if pl.Path(image_path).suffix.lower() not in extensions:
        raise ValueError(f"unsupported file type: {pl.Path(image_path).suffix}")
    if not pl.Path(image_path).exists():
        raise ValueError(f"Image path does not exist: {pl.Path(image_path)}")
    return True

# returns the info image depending on whether grayscale or rbg,
# image is a NumPy array automatically created from imread() in load_image()
# when imread() reads the image it creates and array
def get_image_info(image):
    height = 0
    width = 0
    channels = 0
    # have to check is grayscale or rgb because rgb has channels in its shape (its 3D not 2D data),
    # so will get an unpack error if you don't account for both
    if len(image.shape) == 2:
        height,width = image.shape
        channels = 1
    elif len(image.shape) == 3:
        height, width, channels = image.shape
    # returns a dict of the info
    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
    }

# resizes image, takes max_dimension as 1024 by default
def resize_to_max(image, max_dimension=1024):
    height, width = image.shape[:2] # takes only height and width, we won't want channels at all
    # I don't initialize a scale variable at first because pycharm doesn't like it
    if width >= height:
        scale = max_dimension / width
    else:
        scale = max_dimension / height

    new_width = int(scale * width)
    new_height = int(scale * height)
    # interpolation is the method CV uses to figure out what color to make each pixel when resizing
    resized = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA) # returns resized image
    return resized # returns image again, but resized

# saves image to new path
def save_image(image, output_path):
    success = cv.write(output_path, image)

    if not success:
        raise IOError(f"Save Unsuccessful {output_path}")
    else:
        return output_path
