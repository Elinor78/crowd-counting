import os
import sys
import numpy as np 
import cv2
import random
import scipy.io as sio 
import theano
import math

class DataReader():
    def __init__(self, img_path, gt_path, do_shuffle=False):
        self.img_path = img_path
        self.gt_path = gt_path
        self.filenames = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.endswith('.jpg')]
        if not len(self.filenames) > 0:
            sys.exit("No jpg files in {}".format(self.img_path))

        if do_shuffle:
            random.seed(11)
            random.shuffle(self.filenames)

        else:
            self.filenames.sort()

    def get_images(self):
        for f in self.filenames:
            img = cv2.imread(os.path.join(self.img_path, f))
            if img is None:
                sys.exit("Can't read image {}".format(f))

            if len(img.shape) > 1: 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            X = img.reshape((1, 1, img.shape[0], img.shape[1])).astype(theano.config.floatX)
            #print self.gt_path
            gt = sio.loadmat(os.path.join(self.gt_path, f.split('.')[0] + '.mat'))['final_gt']
            Y = gt.reshape((1, 1, gt.shape[0], gt.shape[1])).astype(theano.config.floatX)

            yield (X, Y)

    def get_whole_images_st(self):
        for f in self.filenames:
            img = cv2.imread(os.path.join(self.img_path, f))
            if img is None:
                sys.exit("Can't read image {}".format(f))

            if len(img.shape) > 1: 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #print img.shape
            X = img.reshape((1, 1, img.shape[0], img.shape[1])).astype(theano.config.floatX)

            gt = sio.loadmat(os.path.join(self.gt_path, 'GT_' + f.split('.')[0] + '.mat'))
            #print gt
            gt = self._get_st_data_ground_truth(gt)
            #print gt
            gt = self._get_dotmap(X, gt)
            #print gt
            #print gt.shape
            Y = gt.reshape((1, 1, gt.shape[0], gt.shape[1])).astype(theano.config.floatX)
            
            yield (X, Y)

    def get_images_ucsd(self):

        order = [int(f.split('_')[1]) for f in self.filenames]
        order = zip(order, self.filenames)
        order = sorted(order, key=lambda x: x[0])
        for (_, f) in order:
            img = cv2.imread(os.path.join(self.img_path, f))
            if img is None:
                sys.exit("Can't read image {}".format(f))

            if len(img.shape) > 1: 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            X = img.reshape((1, 1, img.shape[0], img.shape[1])).astype(theano.config.floatX)
            #print self.gt_path
            gt = sio.loadmat(os.path.join(self.gt_path, f.split('.')[0] + '.mat'))['final_gt']
            Y = gt.reshape((1, 1, gt.shape[0], gt.shape[1])).astype(theano.config.floatX)

            yield (X, Y)            

    def _get_st_data_ground_truth(self, gt):
        image_info = gt['image_info']
        value = image_info[0,0]
        assert len(value['location']) == 1
        for i in value['location']:
            assert len(i) == 1
            for j in i:
                return j 

    def create_dotmaps(self, gt, img_h, img_w):
        #print "in create_dotmaps"
        #print gt
        d_map = np.zeros((int(img_h), int(img_w)))
        #print d_map

        gt = gt[gt[:, 0] < img_w, :]
        gt = gt[gt[:, 1] < img_h, :]

        for i in range(gt.shape[0]):
            x = int(max(1, math.floor(gt[i, 0]))) - 1
            y = int(max(1, math.floor(gt[i, 1]))) - 1
            d_map[y, x] = 1.0
        return d_map


    def _get_dotmap(self, img, gt):
        #print img.shape
        d_map_h = math.floor(math.floor(float(img.shape[2]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[3]) / 2.0) / 2.0)
        #print gt
        d_map = self.create_dotmaps(gt / 4.0, d_map_h, d_map_w)
        #print d_map
        return d_map


'''base = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
    ))

st_A_img = 'ucsd_data/whole/images/'
st_A_gt = 'ucsd_data/whole/gt/'

path_img = os.path.join(base, st_A_img)
path_gt = os.path.join(base, st_A_gt)

r = DataReader(path_img, path_gt, False)

x = r.get_images_ucsd()
print x.next()'''
