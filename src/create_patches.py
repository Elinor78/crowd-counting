import scipy.io as sio
import os
import numpy as np
import PIL.Image as pli
import math
import multiprocessing as mlt 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
    

ST_DATA = "Shanghai Tech Dataset"
UCF_DATA = "UCF CC 50 Dataset"

ST_DATA_CONFIG = {
    'dataset': ST_DATA,
    'img_format': 'IMG_{}.jpg',
    'gt_format': 'GT_IMG_{}.mat',
    'img_ext': 'jpg'
}

UCF_DATA_CONFIG = {
    'dataset': UCF_DATA,
    'img_format': '{}.jpg',
    'gt_format': '{}_ann.mat',
    'img_ext': 'jpg'
}

UCSD_DATA_CONFIG = {
    'img_format': 'vidf1_33_00{}_f{}.png',
    'gt_format': 'vidf1_33_00{}_frame_full.mat',
    'numfiles': 200,
    'img_ext': 'jpg'
}

class DimensionException(Exception):
    pass

class BaseCreatePatches:
    def __init__(self, **kwargs):
        self.img_fold = self.get_full_path(kwargs.pop('img_fold'))
        self.gt_fold = self.get_full_path(kwargs.pop('gt_fold'))
        self.final_img_fold = self.get_full_path(kwargs.pop('final_img_fold'), True)
        self.final_gt_fold = self.get_full_path(kwargs.pop('final_gt_fold'), True)
        self.img_prefix = 'IMG_'
        self.img_format = kwargs.pop('img_format')
        self.gt_format = kwargs.pop('gt_format')
        self.numfiles = self.get_numfiles(kwargs)
        self.img_ext = kwargs.pop('img_ext')

    def get_full_path(self, rel_path, makedir=False):
        directory = os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            rel_path
        )
        if makedir:
            if not os.path.exists(directory):
                os.makedirs(directory)

        return directory 

    def get_numfiles(self, kwargs):
        if "numfiles" in kwargs:
            return kwargs.pop('numfiles')
        else:
            return len([f for f in os.listdir(self.img_fold) if f.endswith('.jpg') and os.path.isfile(os.path.join(self.img_fold, f))])  

    def create_dotmaps(self, gt, img_h, img_w):
        d_map = np.zeros((int(img_h), int(img_w)))

        gt = gt[gt[:, 0] < img_w, :]
        gt = gt[gt[:, 1] < img_h, :]

        for i in range(gt.shape[0]):
            x = int(max(1, math.floor(gt[i, 0]))) - 1
            y = int(max(1, math.floor(gt[i, 1]))) - 1
            d_map[y, x] = 1.0
        return d_map

    def check_dim(self, img):
        if img.ndim != 3:
            if img.ndim == 2:
                img = np.stack((img,)*3, axis=2)
            else:
                raise DimensionException("Image has incorrect dimensions. {}".format(img.shape))
        return img

    def save_gt(self, gt, i, count):
         name = '{}{}_{}.mat'.format(self.img_prefix, i + 1, count)
         sio.savemat(os.path.join(self.final_gt_fold, name), {'final_gt': gt})

    def save_image(self, img, i, count):
        name = '{}{}_{}.{}'.format(self.img_prefix, i + 1, count, self.img_ext)
        img = np.uint8(img)
        img = pli.fromarray(img).save(os.path.join(self.final_img_fold, name))

    def plot_image_tiles(self, index):
        fig = plt.figure()
        count = 1
        for i in range(3):
            for j in range(3):
                a=fig.add_subplot(3,3,count)
                a.set_xticks([])
                a.set_yticks([])
                b = mpimg.imread(
                    os.path.join(self.final_img_fold, 'IMG_{}_{}.{}'.format(index, count, self.img_ext)))
                imgplot = plt.imshow(b)
                count += 1
        plt.subplots_adjust(left=None, bottom=.18, right=None, top=None, wspace=.01, hspace=.001)
        plt.show()

    def plot_dot_tiles(self, index):
        fig = plt.figure()
        count = 1
        for i in range(3):
            for j in range(3):
                a=fig.add_subplot(3,3,count)
                a.set_xticks([])
                a.set_yticks([])
                d = sio.loadmat(
                    os.path.join(self.final_gt_fold, 'IMG_{}_{}.mat'.format(index, count)))
                dt = d['final_gt']
                imgplot = plt.imshow(dt, cmap='gray')
                count += 1
        plt.subplots_adjust(left=None, bottom=.18, right=None, top=None, wspace=.01, hspace=.001)
        plt.show() 


class CreatePatches(BaseCreatePatches):

    def __init__(self, **kwargs):
        super(CreatePatches, self).__init__(**kwargs)
        self.dataset = kwargs.pop('dataset')


    def get_image(self, i):
        img_filename = self.img_format.format(i + 1)
        img_path = os.path.join(self.img_fold, img_filename)
        img = pli.open(img_path)
        img = np.asarray(img, dtype=np.uint8)
        return img

    def _get_st_data_ground_truth(self, i):
        gt_filename = self.gt_format.format(i + 1)
        gt_path = os.path.join(self.gt_fold, gt_filename)
        gt = sio.loadmat(gt_path)
        image_info = gt['image_info']
        value = image_info[0,0]
        assert len(value['location']) == 1
        for i in value['location']:
            assert len(i) == 1
            for j in i:
                return j  

    def _get_ucf_data_ground_truth(self, i):
        gt_filename = self.gt_format.format(i + 1)
        gt_path = os.path.join(self.gt_fold, gt_filename)
        gt = sio.loadmat(gt_path)
        ann_points = gt['annPoints']
        return ann_points

    def get_ground_truth(self, i):
        if self.dataset == ST_DATA:
            return self._get_st_data_ground_truth(i) 

        elif self.dataset == UCF_DATA:
            return self._get_ucf_data_ground_truth(i)

    def _create_test_set(self, i):
        #print(i + 1)
        img = self.get_image(i)
        # moved this out of loop because 3rd dim indexing doesn't work in numpy unless already that shape
        img = self.check_dim(img)
        gt = self.get_ground_truth(i)
        print (gt.shape)

        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

        d_map = self.create_dotmaps(gt / 4.0, d_map_h, d_map_w)

        p_h = int(math.floor(float(img.shape[0]) / 3.0))
        p_w = int(math.floor(float(img.shape[1]) / 3.0))
        d_map_ph = int(math.floor(math.floor(p_h / 2.0) / 2.0))
        d_map_pw = int(math.floor(math.floor(p_w / 2.0) / 2.0))
        
        py = 0
        py2 = 0
        count = 1

        for j in range(3):
            px = 0
            px2 = 0
            for k in range(3):
                final_image = img[py:py + p_h, px: px + p_w, :]
                final_gt = d_map[py2: py2 + d_map_ph, px2: px2 + d_map_pw]                 
                px = px + p_w 
                px2 = px2 + d_map_pw
                self.save_image(final_image, i, count)
                self.save_gt(final_gt, i, count)
                count += 1
            py = py + p_h
            py2 = py2 + d_map_ph  

    def create_test_set(self):
        p = mlt.Pool(mlt.cpu_count())
        p.map(self._create_test_set, range(self.numfiles))

        #for i in range(self.numfiles):
            #self._create_test_set(i)


class CreatePatchesUCSD(BaseCreatePatches):
    def __init__(self, **kwargs):
        super(CreatePatchesUCSD, self).__init__(**kwargs)


    def get_image(self, i, j):
        img_filename = self.img_format.format(i, f'{(j + 1):03}')
        #print (img_filename)
        img_path = os.path.join(self.sub_img_fold, img_filename)
        img = pli.open(img_path)
        img = np.asarray(img, dtype=np.uint8)
        return img

    def get_ground_truth(self, j):
        gt = self.gts[j]
        for x in gt['loc']:
            for y in x:
                return np.delete(y, 2, 1)

    def _create_test_set(self, i, j):
        print (self.overall_count)
        #print(j + 1)
        img = self.get_image(i, j) # ten image folders, each with 200 images
        # moved this out of loop because 3rd dim indexing doesn't work in numpy unless already that shape
        img = self.check_dim(img)
        gt = self.get_ground_truth(j) # ten frame .mat files, each with 200 locations
        #print (gt.shape)

        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

        d_map = self.create_dotmaps(gt / 4.0, d_map_h, d_map_w)
        #print(np.count_nonzero(d_map))

        p_h = int(math.floor(float(img.shape[0]) / 3.0))
        p_w = int(math.floor(float(img.shape[1]) / 3.0))
        d_map_ph = int(math.floor(math.floor(p_h / 2.0) / 2.0))
        d_map_pw = int(math.floor(math.floor(p_w / 2.0) / 2.0))
        
        py = 0
        py2 = 0
        count = 1

        for _ in range(3):
            px = 0
            px2 = 0
            for __ in range(3):
                final_image = img[py:py + p_h, px: px + p_w, :]
                final_gt = d_map[py2: py2 + d_map_ph, px2: px2 + d_map_pw]                 
                px = px + p_w 
                px2 = px2 + d_map_pw
                self.save_image(final_image, self.overall_count, count)
                self.save_gt(final_gt, self.overall_count, count)
                count += 1
            py = py + p_h
            py2 = py2 + d_map_ph  

        self.overall_count += 1

    def get_gts(self, i):
        gt = sio.loadmat(os.path.join(self.gt_fold, self.gt_format.format(i)))
        return gt['frame'][0]

    def create_test_set(self):
        #p = mlt.Pool(mlt.cpu_count())
        #p.map(self._create_test_set, range(self.numfiles))
        self.overall_count = 0
        for i in range(10):
            self.sub_img_fold = os.path.join(self.img_fold, 'vidf1_33_00{}.y/'.format(i))
            self.gts = self.get_gts(i)

            for j in range(self.numfiles):
                self._create_test_set(i, j)

if __name__ == '__main__':

    inputs = {
        'img_fold': 'ST_DATA/A/test/images/',
        'gt_fold': 'ST_DATA/A/test/ground_truth/',
        'final_img_fold': 'st_data_A_test/images/',
        'final_gt_fold': 'st_data_A_test/gt/'

    }
    inputs.update(**ST_DATA_CONFIG)
    test = CreatePatches(**inputs)
    test.create_test_set()
    test.plot_image_tiles(2)
    test.plot_dot_tiles(2)

    inputs = {
        'img_fold': 'ST_DATA/A/train/images/',
        'gt_fold': 'ST_DATA/A/train/ground_truth/',
        'final_img_fold': 'st_data_A_train/images/',
        'final_gt_fold': 'st_data_A_train/gt/'

    }
    inputs.update(**ST_DATA_CONFIG)
    test = CreatePatches(**inputs)
    test.create_test_set()
    test.plot_image_tiles(2)
    test.plot_dot_tiles(2)

    
    inputs = {
        'img_fold': 'ST_DATA/B/test_data/images/',
        'gt_fold': 'ST_DATA/B/test_data/ground_truth/',
        'final_img_fold': 'st_data_B_test/images/',
        'final_gt_fold': 'st_data_B_test/gt/'

    }
    inputs.update(**ST_DATA_CONFIG)
    test = CreatePatches(**inputs)
    test.create_test_set()
    test.plot_image_tiles(2)
    test.plot_dot_tiles(2)

    inputs = {
        'img_fold': 'ST_DATA/B/train_data/images/',
        'gt_fold': 'ST_DATA/B/train_data/ground_truth/',
        'final_img_fold': 'st_data_B_train/images/',
        'final_gt_fold': 'st_data_B_train/gt/'

    }
    inputs.update(**ST_DATA_CONFIG)
    test = CreatePatches(**inputs)
    test.create_test_set()
    test.plot_image_tiles(2)
    test.plot_dot_tiles(2)    

  
    inputs = {
        'img_fold': 'UCF_CC_50/',
        'gt_fold': 'UCF_CC_50/',
        'final_img_fold': 'ucf_data/images/',
        'final_gt_fold': 'ucf_data/gt/',

    }
    inputs.update(**UCF_DATA_CONFIG)

    test = CreatePatches(**inputs)
    test.create_test_set()
    test.plot_image_tiles(1)
    test.plot_dot_tiles(1)

 
    
    inputs = {
        'img_fold': 'ucsdpeds/vidf/',
        'gt_fold':  'gt_1_33/',
        'final_img_fold': 'ucsd_data/images/',
        'final_gt_fold': 'ucsd_data/gt/'
    }
    inputs.update(**UCSD_DATA_CONFIG)
    test = CreatePatchesUCSD(**inputs)
    test.create_test_set()
    test.plot_image_tiles(40)
    test.plot_dot_tiles(40)
    

    
















