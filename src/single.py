from get_data import DataReader
import os
import itertools
from networks import *
from utils import load_nets


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

def run_single_patches(path_img, path_gt):
	r = DataReader(path_img, path_gt, False)

	output_diffs = []
	regressor_diffs = []

	for i, (X, Y) in enumerate(r.get_images()):
		gt_crowd_count = Y.sum()
		# determine which regressor should be used
		regs_out = []
		for reg in run_funcs[1:]:
			regs_out.append(gt_crowd_count - reg(X).sum())
		gt_regressor = np.argmin(np.abs(regs_out))

		# switch and regressor output
		switch_output = run_funcs[0](X)  
		regressor = np.argmax(switch_output, axis = 1)[0]
		regressor_output = run_funcs[regressor + 1](X)

		#diffs
		output_diffs.append(np.abs(gt_crowd_count - regressor_output.sum()))
		regressor_diffs.append(gt_regressor == regressor)


	output_diffs = np.array(output_diffs)
	regressor_diffs = np.array(regressor_diffs)

	mae_patches = output_diffs.sum()/output_diffs.size
	mse_patches = ((output_diffs*output_diffs).sum()/output_diffs.size)**.5

	#add up diffs for image (9 patches)
	reducer = np.array(range(0,output_diffs.size,9))
	output_diffs = np.add.reduceat(output_diffs, reducer)
	mae_images = output_diffs.sum()/output_diffs.size
	mse_images = ((output_diffs*output_diffs).sum()/output_diffs.size)**.5

	switch_accuracy = float(regressor_diffs[regressor_diffs == True].size)/float(regressor_diffs.size)

	print "MAE for patches: {}".format(mae_patches)
	print "MSE for patches: {}".format(mse_patches)
	print "MAE for images: {}".format(mae_images)
	print "MSE for images: {}".format(mse_images)
	print "Switch Accuracy: {}".format(switch_accuracy)


def run_single_whole(path_img, path_gt):
	r = DataReader(path_img, path_gt, False)

	output_diffs = []
	regressor_diffs = []

	for i, (X, Y) in enumerate(r.get_whole_images_st()):
		gt_crowd_count = Y.sum()
		# determine which regressor should be used
		regs_out = []
		for reg in run_funcs[1:]:
			regs_out.append(gt_crowd_count - reg(X).sum())
		gt_regressor = np.argmin(np.abs(regs_out))

		# switch and regressor output
		switch_output = run_funcs[0](X)  
		regressor = np.argmax(switch_output, axis = 1)[0]
		regressor_output = run_funcs[regressor + 1](X)

		#diffs
		output_diffs.append(np.abs(gt_crowd_count - regressor_output.sum()))
		regressor_diffs.append(gt_regressor == regressor)

		#if i == 2:
		#	break

	output_diffs = np.array(output_diffs)
	regressor_diffs = np.array(regressor_diffs)

	mae_images = output_diffs.sum()/output_diffs.size
	mse_images = ((output_diffs*output_diffs).sum()/output_diffs.size)**.5

	switch_accuracy = float(regressor_diffs[regressor_diffs == True].size)/float(regressor_diffs.size)

	print "MAE for images: {}".format(mae_images)
	print "MSE for images: {}".format(mse_images)
	print "Switch Accuracy: {}".format(switch_accuracy)	

if __name__ == "__main__":
	'''
	Data was run on the paths given here. Data not included in repo, but is available upon request.
	'''
	run_funcs = run_scnn()

	base = os.path.dirname(os.path.dirname(
		os.path.abspath(__file__)
		))

	st_A_img = 'st_data_A_test/images/'
	st_A_gt = 'st_data_A_test/gt/'

	path_img = os.path.join(base, st_A_img)
	path_gt = os.path.join(base, st_A_gt)
	print "run_single_patches on Shanhai Tech Test Data Part A"
	run_single_patches(path_img, path_gt)


	st_A_img = 'ST_DATA/A/test/images/'
	st_A_gt = 'ST_DATA/A/test/ground_truth/'

	path_img = os.path.join(base, st_A_img)
	path_gt = os.path.join(base, st_A_gt)
	print "run_single_whole on Shanhai Tech Test Data Part A"
	run_single_whole(path_img,path_gt)

	st_B_img = 'st_data_B_test/images/'
	st_B_gt = 'st_data_B_test/gt/'	

	path_img = os.path.join(base, st_B_img)
	path_gt = os.path.join(base, st_B_gt)

	print "run_single_patches on Shanhai Tech Test Data Part B"
	run_single_patches(path_img, path_gt)


	st_B_img = 'ST_DATA/B/test_data/images/'
	st_B_gt = 'ST_DATA/B/test_data/ground_truth/'

	path_img = os.path.join(base, st_B_img)
	path_gt = os.path.join(base, st_B_gt)
	print "run_single_whole on Shanhai Tech Test Data Part B"
	run_single_whole(path_img,path_gt)

	ucf_img = 'ucf_data/images/'
	ucf_gt = 'ucf_data/gt/'

	path_img = os.path.join(base, ucf_img)
	path_gt = os.path.join(base, ucf_gt)
	print "run_single_patches on UCF Data"
	run_single_patches(path_img, path_gt)

	ucf_img = 'ucf_data/whole/images/'
	ucf_gt = 'ucf_data/whole/gt/'

	path_img = os.path.join(base, ucf_img)
	path_gt = os.path.join(base, ucf_gt)
	print "Run Whole on UCF Data"
	run_single_patches(path_img, path_gt)	# data already in correct format for run_single_patches, but is whole image

	ucsd_img = "ucsd_data/images/"
	ucsd_gt = "ucsd_data/gt/"

	path_img = os.path.join(base, ucsd_img)
	path_gt = os.path.join(base, ucsd_gt)
	print "run_single_patches on UCSD Data"
	run_single_patches(path_img, path_gt)

	ucsd_img = "ucsd_data/whole/images/"
	ucsd_gt = "ucsd_data/whole/gt/"

	path_img = os.path.join(base, ucsd_img)
	path_gt = os.path.join(base, ucsd_gt)
	print "Run Whole on UCSD Data"
	run_single_patches(path_img, path_gt) # data already in correct format for run_single_patches, but is whole image		




