import cv2
import os

img_dir = r'data\demo'
name_list = os.listdir(img_dir)

for name in name_list:
    img_path = os.path.join(img_dir, name)

    img = cv2.imread(img_path)

    img = cv2.resize(img, (256, 256))

    cv2.imwrite(img_path, img)
