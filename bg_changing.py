from rembg import remove
from PIL import Image
import io
import cv2
import numpy as np


with open('quiz_image.png', 'rb') as i:
    input_image = i.read()
output_image = remove(input_image)
image = Image.open(io.BytesIO(output_image)).convert("RGBA")
white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
final = Image.alpha_composite(white_bg, image).convert("RGB")
final_np = np.array(final)
final_img = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
cv2.imshow("Image with white background", final_img)
cv2.waitKey(0)


