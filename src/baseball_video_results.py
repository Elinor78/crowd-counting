import json
import numpy as np 
def get_baseball_regressors():
	filenames = ["exp_outputs/baseball/test_video_algorithm_{}_json.json".format(x) for x in range(1,6)]
	regressors = {}
	patch_counts = {}
	total_counts = {}
	for i, filename in enumerate(filenames):
		print "reading"
		with open(filename, 'r') as f:
			print "loading"
			d = json.load(f)
			print "appending"
			regressors[i + 1] = d['all_regressors']
			total_counts[i + 1] = d['all_total_counts']
			if "all_patch_counts" in d:
				patch_counts[i + 1] = d['all_patch_counts']

	with open("exp_outputs/baseball/regressors.json", 'wb') as ff:
		json.dump(regressors , ff)
	with open("exp_outputs/baseball/patch_counts.json", 'wb') as ff:
		json.dump(patch_counts , ff)
	with open("exp_outputs/baseball/total_counts.json", 'wb') as ff:
		json.dump(total_counts , ff)


def examine_regressors():
	with open("exp_outputs/baseball/regressors.json", 'r') as f:
		results = json.load(f)

	ordered = {}
	for key in results:
		print key
		ordered[key] = {
			0:0,
			1:0,
			2:0
		}
		for i in results[key]:
			if isinstance(i, list):
				for j in i:
					ordered[key][j] +=1
			else:
				ordered[key][i] +=1

	print ordered
				
def examine_patch_counts():
	with open("exp_outputs/baseball/patch_counts.json", 'r') as f:
		results = json.load(f)	

	for key in results:
		print key
		A = np.array(results[key])
		for i in range(9):
			ave = A[:,i].sum()/A[:,i].size
			_min = A[:,i].min()
			_max = A[:, i].max()
			print "{} & {}".format(_min, _max)

def examine_total_counts():
	with open("exp_outputs/baseball/total_counts.json", 'r') as f:
		results = json.load(f)	
	t = []
	for i in range(1,6):
		t.append(results[str(i)])

	A = np.array(t)

	assert np.array_equal(A[0,:], A[1,:])
	assert np.array_equal(A[1,:], A[2,:])
	assert np.array_equal(A[3,:], A[4,:])
	print "Total count output equal for algorithms 1,2,3 across all frames"
	print "Total count output equal for algorithms 4,5 across all frames"
	patched = A[0,:]
	whole = A[3,:]
	diffs = whole - patched
	assert diffs[diffs >=0].size == diffs.size - 1
	assert diffs[diffs < 0].size == 1
	print "Whole image total count greater than patched in all frames except one."
	diffs = np.abs(diffs)
	print "Max difference is {}".format(diffs.max())
	print "Min difference is {}".format(diffs.min())
	print "Max total crowd count (patched) is {}".format(patched.max())
	print "Min total crowd count (patched) is {}".format(patched.min())
	print "Max total crowd count (whole) is {}".format(whole.max())
	print "Min total crowd count (whole) is {}".format(whole.min())
	print "Taking algorithm 1 as pseudo ground truth"
	print "MSE: {}".format(((diffs*diffs).sum()/diffs.size)**.5)
	print "MAE: {}".format(diffs.sum()/diffs.size)
	#print diffs
	#print diffs*diffs





examine_total_counts()


