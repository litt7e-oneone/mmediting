# get crop defect image from source image

import json
import os
import numpy as np
import cv2
import base64


def image_to_base64(image_np):
    """
    convert img np.array to base64
    """
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


def read_json_file(json_path):
    with open(json_path, encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def dump_json_file(json_path, json_data):
    json_fp = open(json_path, 'w', encoding="UTF-8")
    json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
    json_fp.write(json_str)
    json_fp.close()


def cv_imread(file_path, format='BGR', dtype='uint8'):
    """
    same function as cv2.imread.
    In case that file_path including Chinese.
    if file_path does not exist, return None
    Args:
        file_path: str, path to img file
        dtype: str 'uint8'  'uint16'
        format: one of 'GRAY', 'RGB', 'BGR', 'BGRA'
    """
    assert format in ['GRAY', 'RGB', 'BGR', 'BGRA']
    color_dict = {'GRAY': 0, 'BGR':1, 'RGB':1, 'BGRA': -1}
    flag = color_dict[format]
    cv_img = None
    # check file_path
    if os.path.exists(file_path):
        try:
            if dtype == 'uint8':
                dtype = np.uint8
            else:
                dtype = np.uint16
            buffer = np.fromfile(file_path, dtype)
            cv_img = cv2.imdecode(buffer, flag)
            if format == 'RGB':  
                cv_img=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except:
            print(r"can't read file", file_path) 
    else:
        print("file_path ", file_path, " does not exist!")
    return cv_img


def cv_imwrite(file_path, data, splix='.jpg'):
    cv2.imencode(splix, data)[1].tofile(file_path)


if '__main__' == __name__:
    json_data = read_json_file(r"D:\ForkRepo\mmediting\private_projects\mask_singan\dataset\source_img.json")

    img = cv_imread(r"D:\ForkRepo\mmediting\private_projects\mask_singan\dataset\source_img.jpg")

    shapes = json_data['shapes']
    shape = shapes[1]
    x1, y1, x2, y2 = int(shape['points'][0][0]), int(shape['points'][0][1]), int(shape['points'][1][0]), int(shape['points'][1][1])

    crop_img = img[y1:y2, x1:x2, :]
    defect_shape = shapes[0]
    points = defect_shape["points"]
    for i in range(len(points)):
        points[i][0] -= x1
        points[i][1] -= y1

    json_data["imagePath"] = "crop_img.jpg"
    json_data["imageData"] = image_to_base64(crop_img)
    json_data["shapes"] = [defect_shape]

    json_data["imageHeight"] =  crop_img.shape[0]
    json_data["imageWidth"] =  crop_img.shape[1]

    cv_imwrite(r"D:\ForkRepo\mmediting\private_projects\mask_singan\dataset\crop_img.jpg", crop_img)
    dump_json_file(r"D:\ForkRepo\mmediting\private_projects\mask_singan\dataset\crop_img.json", json_data)