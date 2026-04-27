import cv2 as cv
import stage1_capture.photo_loader as pl
import stage1_capture.cleaner as cl
import stage2_edges.edge_extractor as ed
import stage2_edges.preprocessor as pp
from stage3_ai.controlnet_pipeline import generate


# stage 1
img = pl.load_image(r"C:\Users\dylan\Downloads\hqdefault.jpg", "rgb")  # ← edit this
img = pl.resize_to_max(img, 1024)
img = cl.clean_sketch(img)

# stage 2
img = ed.run_edge_pipeline(img)
img = pp.run_preprocessor(img)

# stage 3
result = generate(
    edge_map=img,
    prompt="highly detailed ink sketch, sharp dramatic lines, heavy crosshatching, bold contrast, deep shadows, "
           "renaissance drawing style, charcoal and ink, intricate linework, etching style, black and"
           " white with gold accent lines"
)

# display with PIL instead, dont need imshow() all that BS
result.show()