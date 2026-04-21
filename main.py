import cv2 as cv
import stage1_capture.photo_loader as pl
import stage1_capture.cleaner as cl
import stage2_edges.edge_extractor as ed

image = pl.load_image(r"C:\Users\dylan\Downloads\image (1).png", "rgb")

print(pl.get_image_info(image))
image = cl.clean_sketch(image)
print(pl.get_image_info(image))

image = ed.run_edge_pipeline(image)

cv.imshow("Display Window", image)
cv.waitKey(0)
cv.destroyAllWindows()