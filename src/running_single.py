import os
import sys
from networks import *
from utils import load_nets
import cv2
from matplotlib import pyplot as plt
import math
import time
import numpy as np
from scipy import io as sio
import PIL.Image as pli

def prep_frame(img):
    
    count = 1
    images = []

    p_h = int(math.floor(float(img.shape[0]) / 3.0))
    p_w = int(math.floor(float(img.shape[1]) / 3.0))

    # create non-overlapping patches of images 
    py = 1
    for _ in range(3):
        px = 1
        for __ in range(3):
            final_image = img[py: py + p_h - 1, px: px + p_w - 1, :].astype(float)
            px = px + p_w
            #print final_image.shape
            #if size(final_image, 3) < 3
             #   final_image = repmat(final_image, [1, 1, 3]);
            #end
            images.append(final_image)
            
            
        py = py + p_h
    return images

def run_scnn():
    dirname = os.path.dirname
    src = dirname(os.path.abspath(__file__))
    models = os.path.join(src, 'models/coupled_train')

    trained_model_files =   [
                            os.path.join(models, 'deep_patch_classifier.pkl'),
                            os.path.join(models, 'shallow_9x9.pkl'),
                            os.path.join(models, 'shallow_7x7.pkl'),
                            os.path.join(models, 'shallow_5x5.pkl'),
                            ]

    networks =  [
                    deep_patch_classifier(),
                    shallow_net_9x9(), 
                    shallow_net_7x7(), 
                    shallow_net_5x5()
                ]
    
    load_nets(trained_model_files, networks)
    train_funcs, test_funcs, run_funcs = create_network_functions(networks)
    return run_funcs

def plot_image_tiles(images):
    fig = plt.figure()
    count = 0
    for i in range(3):
        for j in range(3):
            a=fig.add_subplot(3,3,count + 1)
            a.set_xticks([])
            a.set_yticks([])
            b = images[count]
            b = b.reshape(b.shape[2], b.shape[3])
            print b.shape
            plt.imshow(b)
            count += 1
    plt.subplots_adjust(left=None, bottom=.18, right=None, top=None, wspace=.01, hspace=.001)
    plt.show()

def stack_patches(images):
    new_horiz = [] 
    vert = []
    for i in range(10):
        if i % 3 == 0 and i != 0:
            vert.append(np.hstack(new_horiz))
            new_horiz = []
            if i == 9:
                break
            print images[i].shape
        new_horiz.append(images[i])
            
    return np.vstack(vert)

def most_common(lst):
    return max(set(lst), key=lst.count)

def output_patched(images):
    _images = []
    for i in images:
        i = i.astype('uint8')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        _images.append(np.pad(i, (20,20), mode='constant'))
    patched_images = stack_patches(_images).astype('uint8')

    return patched_images

def run_single_image(run_funcs, input_filename, output_filename):
    img = cv2.imread(input_filename)
    images = prep_frame(img)
    out = []
    count_out = []
    counts = 0
    for i in images:
        i = i.astype('uint8')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i = i.reshape((1, 1, i.shape[0], i.shape[1]))
        i = i.astype(theano.config.floatX)  
        switch_output = run_funcs[0](i)   
        regressor = np.argmax(switch_output, axis = 1)[0] 
        regressor_output = run_funcs[regressor + 1](i)
        regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3]) 
        out.append(regressor_output) 
        patch_count = regressor_output.sum()
        counts += patch_count
        blank_patch = np.zeros_like(regressor_output).astype('uint8')
        cv2.putText(blank_patch,"{}".format(int(round(patch_count))), (blank_patch.shape[0]/3,blank_patch.shape[1]/3), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        count_out.append(blank_patch)

    final = stack_patches(out)
    final_copy = final.copy()
    final_copy = cv2.normalize(final, dst=final_copy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    im_color = cv2.applyColorMap(final_copy, cv2.COLORMAP_JET)    
    

    final_counts = stack_patches(count_out)

    counts = int(round(counts))
    cv2.putText(img,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
    
    patched_images = output_patched(images)  

    cv2.imwrite(output_filename, im_color)
    f = '{}_count.jpg'.format(output_filename.split('.')[0])
    cv2.imwrite(f, final_counts)
    f = '{}_frame.jpg'.format(output_filename.split('.')[0])
    cv2.imwrite(f, img)
    f = '{}_img_patch.jpg'.format(output_filename.split('.')[0])
    cv2.imwrite(f, patched_images)