import cv2 as cv

# this sorts edges, camera noise and paper have low thresholds
# thresholds are the gradients at every pixel (rate of color change)
def detect_edges(image, threshold1=100, threshold2=200):
    # want to keep a ratio of 1:2 or 1:3 for which edges are rejected and accepted
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale before edge detection")
    # .canny() calculates the gradient and sorts them into a binary image
    # edges are white (255), everything else is black(0)
    edges = cv.Canny(image, threshold1, threshold2)
    return edges

# further sorts edges to make them bolder by using a kernel (always square)
def enhance_edges(image, kernel_size=3):
    # getStructuringElement just creates a kernel, MORPH_RECT makes it rectangular (how it cuts the corners)
    # getStructuringElement requires a shape argument, so need RECT. Can be different though, as ELLIPSE exists
    # these shape options would alter the corner shapes, we just want sharp 4 corners with rectangle (or square)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    # make edge lines thicker with .dilate()
    enhanced = cv.dilate(image, kernel) # dilate expands every white pixel outward by kernel size
    return enhanced

def run_edge_pipeline(image):
    image = detect_edges(image)
    image = enhance_edges(image)
    return image
