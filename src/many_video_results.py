import json
import numpy as np 
import matplotlib.pyplot as plt
def get_many_regressors():
	filenames = ["exp_outputs/many/test_video_algorithm_{}_json.json".format(x) for x in range(1,6)]
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

	with open("exp_outputs/many/regressors.json", 'wb') as ff:
		json.dump(regressors , ff)
	with open("exp_outputs/many/patch_counts.json", 'wb') as ff:
		json.dump(patch_counts , ff)
	with open("exp_outputs/many/total_counts.json", 'wb') as ff:
		json.dump(total_counts , ff)


def examine_regressors():
	with open("exp_outputs/many/regressors.json", 'r') as f:
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
	print "Algorithm & 9X9 & 7X7 & 5X5 \\\\ \hline"
	for key in ordered:
		print "{} & {} & {} & {} \\\\ \hline".format(key, ordered[key][0], ordered[key][1], ordered[key][2])

def most_common(lst):
    return max(set(lst), key=lst.count)

def examine_regressors_per_patch():
	with open("exp_outputs/many/regressors.json", 'r') as f:
		results = json.load(f)

	alg_1 = results['1']
	alg_1 = np.array(alg_1)
	alg_1_ordered = {k:{0:0, 1:0, 2:0} for k in range(9)}
	
	for i in range(9):
		regressors = alg_1[:,i]
		r, counts = np.unique(regressors, return_counts=True)
		for index, item in enumerate(r):
			alg_1_ordered[i][item] = counts[index]
	fig = plt.figure()
	for key in alg_1_ordered:
		ax = fig.add_subplot('33{}'.format( key + 1))
		
		ax.bar(alg_1_ordered[key].keys(), alg_1_ordered[key].values(), color=['r', 'b', 'g'])
		x_ticks = np.append(ax.get_xticks(), alg_1_ordered[key].keys())
		ax.set_xticks(alg_1_ordered[key].keys())
		y = [i for i in alg_1_ordered[key].values() if i != 0]
		ax.set_yticks(y)

	total_frames = len(alg_1)
	plt.show()
	alg_3 = results['3'][0]
	fig = plt.figure()
	for i, item in enumerate(alg_3):
		ax = fig.add_subplot('33{}'.format( i + 1))
		x = [0,1,2]
		y = [0]*3
		y[item] = total_frames
		ax.bar(x,y, color=['r', 'b', 'g'])
		ax.set_xticks(x)
		ax.set_yticks([total_frames])
	plt.show()

	alg_2 = results['2'][0]
	reg = most_common(alg_2)
	fig = plt.figure()
	for i in range(9):
		ax = fig.add_subplot('33{}'.format( i + 1))
		x = [0,1,2]
		y = [0]*3
		y[reg] = total_frames
		ax.bar(x,y, color=['r', 'b', 'g'])
		ax.set_xticks(x)
		ax.set_yticks([total_frames])
	plt.show()
	alg_4 = results['4']
	r, counts = np.unique(alg_4, return_counts=True)
	print r 
	print counts
	alg_5 = results['5']
	r, counts = np.unique(alg_5, return_counts=True)
	print r 
	print counts


				
def examine_patch_counts():
	with open("exp_outputs/many/patch_counts.json", 'r') as f:
		results = json.load(f)	


	t = []
	for i in range(1,4):
		t.append(results[str(i)])
	A = np.array(t)
	print A.shape
	#alg, frame, patch
	alg_1 = 0

	for i in range(9):
		#print "Patch {}".format(i + 1)
		alg_1 = A[0,:,i]
		alg_2 = A[1,:,i]
		alg_3 = A[2,:,i]

		two_diff = np.abs(alg_2 - alg_1)
		three_diff = np.abs(alg_3 - alg_1)
		two_mse = round(((two_diff*two_diff).sum()/two_diff.size)**.5, 1)
		two_mae = round(two_diff.sum()/two_diff.size, 1)
		three_mse = round(((three_diff*three_diff).sum()/three_diff.size)**.5, 1)
		three_mae = round(three_diff.sum()/three_diff.size, 1)
		print "{} & X & {} & {} & X & {} & {} \\\\ \hline".format(i + 1, two_mae, three_mae, two_mse, three_mse)

	print
	for i in range(9):
		#print "Patch {}".format(i + 1)
		alg_1 = A[0,:,i]
		alg_2 = A[1,:,i]
		alg_3 = A[2,:,i]
		a1 = round(alg_1.sum()/alg_1.size, 3)
		a2 = round(alg_2.sum()/alg_2.size, 3)
		a3 = round(alg_3.sum()/alg_3.size, 3)
		assert a1 == a2 == a3

		mi1 = round(alg_1.min(), 3)
		mi2 = round(alg_2.min(), 3)
		mi3 = round(alg_3.min(), 3)
		assert mi1 == mi2 == mi3
		ma1 = round(alg_1.max(), 3)
		ma2 = round(alg_2.max(), 3)
		ma3 = round(alg_3.max(), 3)
		assert ma1 == ma2 == ma3

		print "{} &{} & {} & {} \\\\ \hline".format(i+1,a1, mi1, ma1)



def examine_total_counts():
	with open("exp_outputs/many/total_counts.json", 'r') as f:
		results = json.load(f)	
	t = []
	for i in range(1,6):
		t.append(results[str(i)])

	A = np.array(t)
	print A.shape

	one = A[0,:]
	two = A[1,:]
	three = A[2,:]
	four = A[3,:]
	five = A[4,:]

	two_diff = np.abs(two - one)
	three_diff = np.abs(three - one)
	four_diff = np.abs(four - one)
	five_diff = np.abs(five - one)

	mae2 = int(round(two_diff.sum()/two_diff.size))
	mae3 = int(round(three_diff.sum()/three_diff.size))
	mae4 = int(round(four_diff.sum()/four_diff.size))
	mae5 = int(round(five_diff.sum()/five_diff.size))
	mse2 = int(round(((two_diff*two_diff).sum()/two_diff.size)**.5))
	mse3 = int(round(((three_diff*three_diff).sum()/three_diff.size)**.5))
	mse4 = int(round(((four_diff*four_diff).sum()/four_diff.size)**.5))
	mse5 = int(round(((five_diff*five_diff).sum()/five_diff.size)**.5))

	a1 = int(round(one.sum()/one.size))
	a2 = int(round(two.sum()/two.size))
	a3 = int(round(three.sum()/three.size))
	a4 = int(round(four.sum()/four.size))
	a5 = int(round(five.sum()/five.size))

	mi1 = int(round(one.min()))
	mi2 = int(round(two.min()))
	mi3 = int(round(three.min()))
	mi4 = int(round(four.min()))
	mi5 = int(round(five.min()))

	ma1 = int(round(one.max()))
	ma2 = int(round(two.max()))
	ma3 = int(round(three.max()))
	ma4 = int(round(four.max()))
	ma5 = int(round(five.max()))

	print "Algorithm & Ave & Min & Max & MAE & MSE"

	print "1 & {} & {} & {} & {} & {} \\\\ \hline".format(a1, mi1, ma1, 'N/A', 'N/A')
	print "2 & {} & {} & {} & {} & {} \\\\ \hline".format(a2, mi2, ma2, mae2, mse2)
	print "3 & {} & {} & {} & {} & {} \\\\ \hline".format(a3, mi3, ma3, mae3, mse3)
	print "4 & {} & {} & {} & {} & {} \\\\ \hline".format(a4, mi4, ma4, mae4, mse4)
	print "5 & {} & {} & {} & {} & {} \\\\ \hline".format(a5, mi5, ma5, mae5, mse5)




examine_total_counts()




