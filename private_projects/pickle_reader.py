import pickle
import numpy as np
import cv2


def read_noise_pkl(pkl_path):
    f = open(pkl_path, 'rb')
    data = pickle.load(f)
    print('curr_stage', data['curr_stage'], " / " , len(data['noise_weights']) - 1)
    noise_weights = data.get('noise_weights', False)
    fixed_noises = data.get('fixed_noises', False)

    if noise_weights and fixed_noises:
        print('index ', 'noise_weights ', 'fixed_noises')
        for i, item in enumerate(zip(noise_weights, fixed_noises)):
            print(i, ' ', item[0], ' ', item[1].shape)
    
        # for i, noise in enumerate(fixed_noises):
        #    cv2.imwrite('noise_{}.jpg'.format(i), noise[0]) 

    else:
        if not noise_weights:
            print('noise_weights key does not exist')
        if not fixed_noises:
            print('fixed_noises key does not exist')


def change_curr_stage(pkl_path, stage=0):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data['curr_stage'] = stage

    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f) 


def construct_nosie_pkl(num_stage, current_stage, fixed_noises):
    return
    

def add_noise_inputs():
    return

if __name__ == '__main__':
    # read_noise_pkl(r'./project/singan/train_detect5/pickle/iter_48001.pkl')  # project\singan\train_detect1\temp.pkl
    # change_curr_stage(r'D:\Megarobo\mmediting\data\singan\singan_fis_20210406_201006-860d91b6.pkl', 10)

    
    # print(pow(48 / 400, 1/8))
    # print(61/ 48 * 250)
    a = 800 
    for i in range(10):
        a *= 0.75
        print(a)