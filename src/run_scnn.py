import os
from networks import *
from utils import load_nets
import cv2
from matplotlib import pyplot as plt
import math
import time
import numpy as np

def prep_frame(img):
    
    count = 1;
    images = []



    p_h = int(math.floor(float(img.shape[0]) / 3.0))
    p_w = int(math.floor(float(img.shape[1]) / 3.0))


    # create non-overlapping patches of images 
    py = 1;
    for _ in range(3):
        px = 1;
        for __ in range(3):
            final_image = img[py: py + p_h - 1, px: px + p_w - 1, :].astype(float)
            px = px + p_w
            #print final_image.shape
            #if size(final_image, 3) < 3
             #   final_image = repmat(final_image, [1, 1, 3]);
            #end
            images.append(final_image)
            
            
        py = py + p_h;
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

def run_cv2(run_funcs):
    cap = cv2.VideoCapture('baseball_crowd2_hd.mp4')

    #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    #cv2.moveWindow('frame', 20, 20)
    #cv2.resizeWindow('frame', 600,400)

    #cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
    #cv2.moveWindow('gray', 700, 20)
    #cv2.resizeWindow('gray', 600,400)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter('output.mp4',fourcc, 20.0, (477,267), isColor=False)

    count = 1
    start = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        images = prep_frame(frame)
        out = []
        for i in images:
            #print i.dtype
            i = i.astype('uint8')
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = i.reshape((1, 1, i.shape[0], i.shape[1]))
            i = i.astype(theano.config.floatX)
            #cv2.imshow('frame', i.astype('uint8'))
            #time.sleep(1)
            switch_output = run_funcs[0](i)
            
            regressor = np.argmax(switch_output, axis = 1)[0]
            
            regressor_output = run_funcs[regressor + 1](i)
            regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
            out.append(regressor_output)

        final = stack_patches(out)

        final_copy = final.copy()
        final_copy = cv2.normalize(final, dst=final_copy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


        writer.write(final_copy)



        
        count += 1
        if count == 10:
            break
        #cv2.imshow('frame',frame)
        #cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    print "Patches: {}".format(end - start)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    img = cv2.imread('IMG_1.jpg')
    plt.imshow(img)
    plt.show()
    images = prep_frame(img)
    plot_image_tiles(images)
    '''


    run_funcs = run_scnn()
    run_cv2(run_funcs)

