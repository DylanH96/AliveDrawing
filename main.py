import cv2 as cv
import stage1_capture.photo_loader as pl
import stage1_capture.cleaner as cl
import stage2_edges.edge_extractor as ed
import stage2_edges.preprocessor as pp
from stage3_ai.controlnet_pipeline import generate
from stage3_ai.interpolator import generate_interpolated

# stage 1
img = pl.load_image(r"C:\Users\dylan\Downloads\hqdefault.jpg", "rgb")  # ← edit this
img = pl.resize_to_max(img)
img = cl.clean_sketch(img)

# stage 2
img_gray = ed.run_edge_pipeline(img)
img_gray = pp.run_preprocessor(img_gray, 1024)

# stage 3
result = generate_interpolated(
    original=img,
    edge_map=img_gray,
    prompt="highly detailed ink sketch, sharp dramatic lines, heavy crosshatching, bold contrast, deep shadows, "
           "renaissance drawing style, charcoal and ink, intricate linework, etching style, black and"
           " white with gold accent lines",
    strength=2.0,
    image_size=1024
)

# display with PIL instead, dont need imshow() all that BS
result.show()