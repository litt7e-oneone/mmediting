import cv2
import numpy as np
from matplotlib import pyplot as plt 


# img = cv2.imread(r'D:\Megarobo\mmediting\data\singan\000000000000200028_4_1_TA07_02_20210903175449378_00_0_1920_800_2720.jpg')  # h, w, 3

# img = cv2.resize(img, (64, 64))

# cv2.imwrite(r'temp.jpg', img)  

def draw_hist_gray(img_gray, bins=255, ranges=[0, 256]):
    """
    直方图
    """
    if isinstance(img_gray, str):
        img_gray = cv2.imread(img_gray, 0)  
    plt.hist(img_gray.ravel(),bins,ranges) 
    plt.title("Matplotlib Method Hist")
    plt.show()


def norm_min_max(img):
    min_v = np.min(img)
    max_v = np.max(img)

    img = (img - min_v) / (max_v - min_v) * 255.
    img = img.astype(np.uint8)
    return img


def crop_img(img_path, x1, y1, x2, y2):
    img_gray = cv2.imread(img_path, 0)
    print(img_gray.shape)
    img_crop = img_gray[y1:y2, x1: x2]
    cv2.imwrite(img_path[:-4] + '-crop.jpg', img_crop)

if __name__ == '__main__':
    # img_gray = cv2.imread(r'000000000000200082_4_2_TA07_02_20210908180649902_00_3040_1280_3840_2080-crop.jpg', 0)  
    # img_gray = norm_min_max(img_gray)
    # cv2.imwrite('00.jpg', img_gray)
    draw_hist_gray(r'00.jpg')
    # draw_hist_gray(r'data\singan\000000000000100320_2_4_TA07_02_20220110100006509_01_640_1280_1440_2080.jpg')
    # img = cv2.imread(r'data\singan\000000000000200028_4_1_TA07_02_20210903175449378_00_0_1920_800_2720.jpg', 0)
    # equ = cv2.equalizeHist(img)
    # cv2.imwrite(r'data\singan\equ.jpg', equ)

    # img = cv2.imread(r'D:\Megarobo\mmediting\data\singan\000000000000200028_4_1_TA07_02_20210903175449378_00_0_1920_800_2720.jpg')  # h, w, 3

    # img = cv2.resize(img, (320, 320))

    # cv2.imwrite(r'D:\Megarobo\mmediting\data\singan\temp.jpg', img)  

    # crop_img(img_path=r'000000000000200082_4_2_TA07_02_20210908180649902_00_3040_1280_3840_2080.jpg', 
    #          x1=0, 
    #          y1=99, 
    #          x2=800, 
    #          y2=223)
    

    # num_scales = 10
    # iters_per_scale = 1000
    # discriminator_steps = 3
    # iters = (num_scales + 1) * iters_per_scale * discriminator_steps
    # print(iters)