import os
import sys
import json
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

def process_patches(run_funcs, images, first, regressors=None):
    regressors_used = []
    out = []
    count_out = []
    counts = []
    for index, i in enumerate(images):
        i = i.astype('uint8')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i = i.reshape((1, 1, i.shape[0], i.shape[1]))
        i = i.astype(theano.config.floatX)

        if first:
            switch_output = run_funcs[0](i)
            regressor = np.argmax(switch_output, axis = 1)[0]
            regressors_used.append(regressor)
        elif not first and regressors is None:
            print "switching (after first)"
            switch_output = run_funcs[0](i)
            regressor = np.argmax(switch_output, axis = 1)[0] 
            regressors_used.append(regressor)
        elif not first and isinstance(regressors, list):
            regressor = regressors[index]
        elif not first and isinstance(regressors, int):
            regressor = regressors


        
        regressor_output = run_funcs[regressor + 1](i)
        regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
        out.append(regressor_output)

        patch_count = regressor_output.sum()
        counts.append(patch_count)
        blank_patch = np.zeros_like(regressor_output).astype('uint8')
        cv2.putText(blank_patch,"{}".format(int(round(patch_count))), (blank_patch.shape[0]/3,blank_patch.shape[1]/3), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        count_out.append(blank_patch)

    return regressors_used, out, count_out, counts 

def output_heatmap(out):  
    final = stack_patches(out)
    final_copy = final.copy()
    final_copy = cv2.normalize(final, dst=final_copy, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    final_copy = cv2.applyColorMap(final_copy, cv2.COLORMAP_JET)

    final_shape = (final_copy.shape[1], final_copy.shape[0])

    return final_copy, final_shape, final  

def output_patched(images):
    _images = []
    for i in images:
        i = i.astype('uint8')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        _images.append(np.pad(i, (20,20), mode='constant'))
    patched_images = stack_patches(_images).astype('uint8')

    return patched_images

def video_algorithm_1(run_funcs, input_filename, output_filename):
    '''
    For each frame in video:
    Patch frame into 9 patches
    for patch in patches:
        regressor = switch(patch)
        output = regressor(patch)
    '''
    output_base = "{}_video_algorithm_1".format(output_filename)
    heat_map_filename = "{}_heatmap.mp4".format(output_base)
    count_filename = "{}_count.mp4".format(output_base)
    frame_filename = "{}_frame.mp4".format(output_base)
    patch_filename = "{}_patches.mp4".format(output_base)
    json_filename = "{}_json.json".format(output_base)

    all_regressors = []
    all_patch_counts = []
    all_total_counts = []
    all_heatmaps = []


    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()  
    images = prep_frame(frame) 
    regressors_used, out, count_out, counts = process_patches(run_funcs, images, True)
    all_regressors.append(regressors_used)
    all_patch_counts.append(counts)
    final_copy, final_shape, final = output_heatmap(out)
    all_heatmaps.append(final)
    final_counts = stack_patches(count_out)
    counts = int(round(sum(counts)))
    all_total_counts.append(counts)
    cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
    patched_images = output_patched(images)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    heatmap_writer = cv2.VideoWriter(heat_map_filename,fourcc, 20.0, final_shape, isColor=True)
    count_writer = cv2.VideoWriter(count_filename,fourcc, 20.0, final_shape, isColor=False)
    frame_writer = cv2.VideoWriter(frame_filename,fourcc, 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    img_patch_writer = cv2.VideoWriter(patch_filename, fourcc, 20.0, (patched_images.shape[1], patched_images.shape[0]), isColor=False)

    heatmap_writer.write(final_copy)
    frame_writer.write(frame)
    count_writer.write(final_counts)    
    img_patch_writer.write(patched_images)
    count = 1

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            images = prep_frame(frame)
            regressors_used, out, count_out, counts = process_patches(run_funcs, images, False, regressors=None)
            all_regressors.append(regressors_used)
            all_patch_counts.append(counts)
            final_counts = stack_patches(count_out)
            final_copy, _, final = output_heatmap(out)
            all_heatmaps.append(final)
            patched_images = output_patched(images)
            counts = int(round(sum(counts)))
            all_total_counts.append(counts)
            cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
            frame_writer.write(frame)
            heatmap_writer.write(final_copy)
            count_writer.write(final_counts)
            img_patch_writer.write(patched_images)



            count += 1
            #if count >= 3:
            #    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    img_patch_writer.release()
    frame_writer.release()
    count_writer.release()
    heatmap_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    output = {
        'all_regressors': all_regressors,
        'all_heatmaps': [x.tolist() for x in all_heatmaps],
        'all_patch_counts': all_patch_counts,
        'all_total_counts': all_total_counts
    }
    with open(json_filename, 'wb') as f:
        json.dump(output, f, indent=2)


    #return all_regressors, all_heatmaps, all_patch_counts, all_total_counts

def video_algorithm_2(run_funcs, input_filename, output_filename):
    '''
    Obtain first frame of video
    Patch frame into 9 patches
    regressors = []
    for patch in patches:
        regressor = switch(patch)
        output = regressor(patch) 
        regressors.append(regressor)
    regressor = most_common(regressors)
    For each frame in video[1:]:
        Patch frame into 9 patches
        for patch in patches:
            output = regressor(patch)
    '''
    output_base = "{}_video_algorithm_2".format(output_filename)
    heat_map_filename = "{}_heatmap.mp4".format(output_base)
    count_filename = "{}_count.mp4".format(output_base)
    frame_filename = "{}_frame.mp4".format(output_base)
    patch_filename = "{}_patches.mp4".format(output_base)
    json_filename = "{}_json.json".format(output_base)

    all_regressors = []
    all_patch_counts = []
    all_total_counts = []
    all_heatmaps = []

    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()
    images = prep_frame(frame)
    regressors_used, out, count_out, counts = process_patches(run_funcs, images, True)
    all_regressors.append(regressors_used)
    all_patch_counts.append(counts)
    regressor = most_common(regressors_used)
    final_copy, final_shape, final = output_heatmap(out)
    all_heatmaps.append(final)
    final_counts = stack_patches(count_out)
    counts = int(round(sum(counts)))
    all_total_counts.append(counts)
    cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
    patched_images = output_patched(images)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    heatmap_writer = cv2.VideoWriter(heat_map_filename,fourcc, 20.0, final_shape, isColor=True)
    count_writer = cv2.VideoWriter(count_filename,fourcc, 20.0, final_shape, isColor=False)
    frame_writer = cv2.VideoWriter(frame_filename,fourcc, 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    img_patch_writer = cv2.VideoWriter(patch_filename, fourcc, 20.0, (patched_images.shape[1], patched_images.shape[0]), isColor=False)
    heatmap_writer.write(final_copy)
    frame_writer.write(frame)
    count_writer.write(final_counts)    
    img_patch_writer.write(patched_images)


    count = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images = prep_frame(frame)
            _, out, count_out, counts = process_patches(run_funcs, images, False, regressors=regressor)
            all_patch_counts.append(counts)
            final_counts = stack_patches(count_out)
            final_copy, _, final = output_heatmap(out)
            all_heatmaps.append(final)
            patched_images = output_patched(images)
            counts = int(round(sum(counts)))
            all_total_counts.append(counts)
            cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
            frame_writer.write(frame)
            heatmap_writer.write(final_copy)
            count_writer.write(final_counts)
            img_patch_writer.write(patched_images)
            count += 1
            #if count >= 3:
            #    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    img_patch_writer.release()
    frame_writer.release()
    count_writer.release()
    heatmap_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    output = {
        'all_regressors': all_regressors,
        'all_heatmaps': [x.tolist() for x in all_heatmaps],
        'all_patch_counts': all_patch_counts,
        'all_total_counts': all_total_counts
    }
    with open(json_filename, 'wb') as f:
        json.dump(output, f, indent=2)

def video_algorithm_3(run_funcs, input_filename, output_filename):
    '''
    Obtain first frame of video
    Patch frame into 9 patches
    regressors = []
    for patch in patches:
        regressor = switch(patch)
        output = regressor(patch) 
        regressors.append(regressor)
    For each frame in video[1:]:
        Patch frame into 9 patches
        for index, patch in patches:
            output = regressors[index](patch)
    '''
    output_base = "{}_video_algorithm_3".format(output_filename)
    heat_map_filename = "{}_heatmap.mp4".format(output_base)
    count_filename = "{}_count.mp4".format(output_base)
    frame_filename = "{}_frame.mp4".format(output_base)
    patch_filename = "{}_patches.mp4".format(output_base)
    json_filename = "{}_json.json".format(output_base)

    all_regressors = []
    all_patch_counts = []
    all_total_counts = []
    all_heatmaps = []

    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()
    images = prep_frame(frame)
    regressors_used, out, count_out, counts = process_patches(run_funcs, images, True)
    all_regressors.append(regressors_used)
    all_patch_counts.append(counts)
    final_copy, final_shape, final = output_heatmap(out)
    all_heatmaps.append(final)
    final_counts = stack_patches(count_out)
    counts = int(round(sum(counts)))
    all_total_counts.append(counts)
    cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
    patched_images = output_patched(images)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    heatmap_writer = cv2.VideoWriter(heat_map_filename,fourcc, 20.0, final_shape, isColor=True)
    count_writer = cv2.VideoWriter(count_filename,fourcc, 20.0, final_shape, isColor=False)
    frame_writer = cv2.VideoWriter(frame_filename,fourcc, 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    img_patch_writer = cv2.VideoWriter(patch_filename, fourcc, 20.0, (patched_images.shape[1], patched_images.shape[0]), isColor=False)
    heatmap_writer.write(final_copy)
    frame_writer.write(frame)
    count_writer.write(final_counts)    
    img_patch_writer.write(patched_images)

    count = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images = prep_frame(frame)
            _, out, count_out, counts = process_patches(run_funcs, images, False, regressors=regressors_used)
            all_patch_counts.append(counts)
            final_counts = stack_patches(count_out)
            final_copy, _, final = output_heatmap(out)
            all_heatmaps.append(final)
            patched_images = output_patched(images)
            counts = int(round(sum(counts)))
            all_total_counts.append(counts)
            cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
            frame_writer.write(frame)
            heatmap_writer.write(final_copy)
            count_writer.write(final_counts)
            img_patch_writer.write(patched_images)
            count += 1
            #if count >= 3:
            #    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    img_patch_writer.release()
    frame_writer.release()
    count_writer.release()
    heatmap_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    output = {
        'all_regressors': all_regressors,
        'all_heatmaps': [x.tolist() for x in all_heatmaps],
        'all_patch_counts': all_patch_counts,
        'all_total_counts': all_total_counts
    }
    with open(json_filename, 'wb') as f:
        json.dump(output, f, indent=2)

def video_algorithm_4(run_funcs, input_filename, output_filename):
    '''
    For each frame in video:
        regressor = switch(frame)
        output = regressor(frame)
    '''
    output_base = "{}_video_algorithm_4".format(output_filename)
    heat_map_filename = "{}_heatmap.mp4".format(output_base)
    frame_filename = "{}_frame.mp4".format(output_base)
    json_filename = "{}_json.json".format(output_base)

    all_regressors = []
    all_total_counts = []
    all_heatmaps = []

    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()  
    frame = cv2.resize(frame, (0,0), fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape((1,1,gray.shape[0],gray.shape[1]))
    gray = gray.astype(theano.config.floatX)
    print 'switching'
    switch_output = run_funcs[0](gray)
    regressor = np.argmax(switch_output, axis=1)[0]
    all_regressors.append(regressor)
    print "regressing"
    regressor_output = run_funcs[regressor + 1](gray)
    regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
    crowd_count = int(round(regressor_output.sum()))
    all_total_counts.append(crowd_count)
    heat_map = regressor_output.copy()
    all_heatmaps.append(heat_map)
    heat_map = cv2.normalize(regressor_output, dst=heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    heat_map_shape = (heat_map.shape[1], heat_map.shape[0])
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    cv2.putText(frame,"{}".format(crowd_count), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)

    print "writing"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    heat_map_writer = cv2.VideoWriter(heat_map_filename,fourcc, 20.0, heat_map_shape, isColor=True)
    frame_writer = cv2.VideoWriter(frame_filename,fourcc, 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    heat_map_writer.write(heat_map)
    frame_writer.write(frame)

    count = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(0,0), fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.reshape((1,1,gray.shape[0],gray.shape[1]))
            gray = gray.astype(theano.config.floatX)
            switch_output = run_funcs[0](gray)
            regressor = np.argmax(switch_output, axis=1)[0]
            all_regressors.append(regressor)
            regressor_output = run_funcs[regressor + 1](gray)
            regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
            crowd_count = int(round(regressor_output.sum()))
            all_total_counts.append(crowd_count)
            heat_map = regressor_output.copy()
            all_heatmaps.append(heat_map)
            heat_map = cv2.normalize(regressor_output, dst=heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            heat_map_shape = (heat_map.shape[1], heat_map.shape[0])
            heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
            cv2.putText(frame,"{}".format(crowd_count), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
            heat_map_writer.write(heat_map)
            frame_writer.write(frame)

            count += 1
            #if count >= 3:
            #    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    frame_writer.release()
    heat_map_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    output = {
        'all_regressors': all_regressors,
        'all_heatmaps': [x.tolist() for x in all_heatmaps],
        'all_total_counts': all_total_counts
    }
    with open(json_filename, 'wb') as f:
        json.dump(output, f, indent=2)

def video_algorithm_5(run_funcs, input_filename, output_filename):
    '''
    Obtain first frame of video
    regressor = switch(frame)
    output = regressor(frame)
    For each frame in video[1:]:
        output = regressor(frame)
    '''
    output_base = "{}_video_algorithm_5".format(output_filename)
    heat_map_filename = "{}_heatmap.mp4".format(output_base)
    frame_filename = "{}_frame.mp4".format(output_base)
    json_filename = "{}_json.json".format(output_base)

    all_regressors = []
    all_total_counts = []
    all_heatmaps = []

    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()  
    #frame = cv2.resize(frame, (0,0), fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape((1,1,gray.shape[0],gray.shape[1]))
    gray = gray.astype(theano.config.floatX)
    print "switching"
    switch_output = run_funcs[0](gray)

    regressor = np.argmax(switch_output, axis=1)[0]
    all_regressors.append(regressor)
    print "regressing"
    regressor_output = run_funcs[regressor + 1](gray)
    regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
    crowd_count = int(round(regressor_output.sum()))
    all_total_counts.append(crowd_count)
    heat_map = regressor_output.copy()
    all_heatmaps.append(heat_map)
    heat_map = cv2.normalize(regressor_output, dst=heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    heat_map_shape = (heat_map.shape[1], heat_map.shape[0])
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    cv2.putText(frame,"{}".format(crowd_count), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)

    print "writing"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    heat_map_writer = cv2.VideoWriter(heat_map_filename,fourcc, 20.0, heat_map_shape, isColor=True)
    frame_writer = cv2.VideoWriter(frame_filename,fourcc, 20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    heat_map_writer.write(heat_map)
    frame_writer.write(frame)



    count = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            #frame = cv2.resize(frame, (0,0), fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.reshape((1,1,gray.shape[0],gray.shape[1]))
            gray = gray.astype(theano.config.floatX)
            switch_output = run_funcs[0](gray)
            #regressor = np.argmax(switch_output, axis=1)[0]
            regressor_output = run_funcs[regressor + 1](gray)
            regressor_output = regressor_output.reshape(regressor_output.shape[2], regressor_output.shape[3])
            crowd_count = int(round(regressor_output.sum()))
            all_total_counts.append(crowd_count)
            heat_map = regressor_output.copy()
            all_heatmaps.append(heat_map)
            heat_map = cv2.normalize(regressor_output, dst=heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            heat_map_shape = (heat_map.shape[1], heat_map.shape[0])
            heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
            cv2.putText(frame,"{}".format(crowd_count), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
            heat_map_writer.write(heat_map)
            frame_writer.write(frame)
            print count
            count += 1
            #if count >= 3:
            #    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    frame_writer.release()
    heat_map_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    output = {
        'all_regressors': all_regressors,
        'all_heatmaps': [x.tolist() for x in all_heatmaps],
        'all_total_counts': all_total_counts
    }
    with open(json_filename, 'wb') as f:
        json.dump(output, f, indent=2)

def images(run_funcs, input_filename, output_filename):
    '''
    For each frame in video:
    Patch frame into 9 patches
    for patch in patches:
        regressor = switch(patch)
        output = regressor(patch)
    '''
    output_base = "{}_video_algorithm_1".format(output_filename)
    heat_map_filename = "{}_heatmap.jpg".format(output_base)
    count_filename = "{}_count.jpg".format(output_base)
    frame_filename = "{}_frame.jpg".format(output_base)
    patch_filename = "{}_patches.jpg".format(output_base)
    print patch_filename
    return


    cap = cv2.VideoCapture(input_filename)
    ret, frame = cap.read()  
    images = prep_frame(frame) 
    regressors_used, out, count_out, counts = process_patches(run_funcs, images, True)
    #all_regressors.append(regressors_used)
    #all_patch_counts.append(counts)
    final_copy, final_shape, final = output_heatmap(out)
    #all_heatmaps.append(final)
    final_counts = stack_patches(count_out)
    counts = int(round(sum(counts)))
    #all_total_counts.append(counts)
    cv2.putText(frame,"{}".format(counts), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,255,0), thickness=7)
    patched_images = output_patched(images)


    cv2.imwrite(frame_filename, frame)
    cv2.imwrite(heat_map_filename, final_copy)
    cv2.imwrite(count_filename, final_counts)
    cv2.imwrite(patch_filename, patched_images)



if __name__ == "__main__":


    assert len(sys.argv) == 3, "Usage python whole.py <input_filename> <output_filename>"
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]


    run_funcs = run_scnn()
    video_algorithm_1(run_funcs, input_filename, output_filename)
    video_algorithm_2(run_funcs, input_filename, output_filename)
    video_algorithm_3(run_funcs, input_filename, output_filename)
    video_algorithm_4(run_funcs, input_filename, output_filename)
    video_algorithm_5(run_funcs, input_filename, output_filename)
    images(run_funcs, input_filename, output_filename)

