import cv2 as cv

# removes noise from image
def denoise(image, strength=10):
    if len(image.shape) == 2:
        denoised = cv.fastNlMeansDenoising(image, h=strength) # for already grayscale
    else:
        denoised = cv.fastNlMeansDenoisingColored(image, h=strength) # for

    return denoised
# makes it grayscale so the computer can analyze the image
def to_grayscale(image):
    if len(image.shape) == 2:
        return image # just returns if already grayscale
    else:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray # makes grayscale if not

# threshold
def threshold(image, thresh_value, max_value):
    if len(image.shape) != 2:
        raise ValueError("Image must be 2d array (grayscale) before thresholding")
    # cv.threshold() returns 2 things: (threshold_value_used, thresholded_image), we only want the first
    _, thresholded = cv.threshold(image, thresh_value, max_value, cv.THRESH_BINARY)
    return thresholded

# implement all 3 methods to clean the image in one go
def clean_sketch(image):
    image = denoise(image)
    image = to_grayscale(image)
    image = threshold(image, thresh_value=100, max_value=255)
    return image