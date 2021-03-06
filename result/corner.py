import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py 
import sys

# dimension of parameter space
Name = ["mix", "flut", "n-pPb", "n-PbPb", "mean", "std", "skew", "Jacobi"]
n_dims = 7+1
# number of model samples
n_samples = 300
# range of parameters
prange = np.zeros([n_dims, 2])
prange[0] = 	[0.0, 1.0]	#dataset mixing parameter
prange[1] = 	[1.0, 5.0]  #fluct
prange[2] = 	[9,    13]	#norm-pPb
prange[3] = 	[1.0,    1.4]	#norm-pPb/norm-PbPb
prange[4] = 	[0,   2.0]	#mean
prange[5] = 	[2.5, 4.5]  #std
prange[6] = [0.0, 2.0]  #skew -- model-1
prange[7] = 	[0.6, 0.9]  #Jacobi

prange2 = np.zeros([n_dims, 2])
prange2[0] = 	[0.0, 1.0]	#dataset mixing parameter
prange2[1] = 	[1.0, 5.0]  #fluct
prange2[2] = 	[9,    13]	#norm-pPb
prange2[3] = 	[1.0,    1.4]	#norm-pPb/norm-PbPb
prange2[4] = 	[0,   2.0]	#mean
prange2[5] = 	[2.5, 4.5]  #std
prange2[6] = [0.0, 0.6]  #skew -- model-2
prange2[7] = 	[0.6, 0.9]  #Jacobi

tb = [5, 5, 5, 5, 5, 5, 5, 5]	# number of ticks  each axis
nbins = [4,4,4,4,4,4,4,4]		# number of tikcs shown, so that adjacent ticks does not overlap...

# loading dataset:
f1 =  h5py.File("./pset-1.hdf5")
f2 =  h5py.File("./pset-2.hdf5")

flat_p1 = f1['pset'].value.T
flat_p2 = f2['pset'].value.T

for i in range(n_dims):
	flat_p1[i] = flat_p1[i]*(prange[i,1]-prange[i,0]) + prange[i,0]
for i in range(n_dims):
	flat_p2[i] = flat_p2[i]*(prange2[i,1]-prange2[i,0]) + prange2[i,0]
flat_p2[3] = flat_p2[2]/flat_p2[3]
flat_p1[3] = flat_p1[2]/flat_p1[3]
prange[3] = 	[8,    12]
prange2[3] = 	[8,    12]
# >>>----------------Corner Plot-------------------------------
#range for plot
prange = np.zeros([n_dims, 2])
prange[0] = 	[0.0, 1.0]	#dataset mixing parameter
prange[1] = 	[2., 4.0]  #fluct
prange[2] = 	[11,  12]	#norm-pPb
prange[3] = 	[9,   10]	#norm-PbPb
prange[4] = 	[0.0,  0.5]	#mean
prange[5] = 	[2.5, 3.0]  #std
prange[6] = 	[0.0, 1.5]  #skew -- model-1
prange[7] = 	[0.6, 0.7]  #Jacobi

plt.figure(figsize = (n_dims*2, n_dims*2))

for i in range(n_dims):
	for j in range(n_dims):
		# Create a subplots
		ax = plt.subplot(n_dims, n_dims, i*n_dims + j + 1)
		
		# Set up axis for this plot
		if i>=j:
			plt.axis('on')
			plt.xticks([])
			plt.yticks([])
			if i==n_dims - 1:
				plt.xticks(np.linspace(prange[j][0], prange[j][1], tb[j]))
				plt.xlabel(Name[j], size = 25)
				ax.xaxis.set_major_locator(MaxNLocator(nbins = nbins[j], prune="upper"))
			if j==0 and i != j:
				plt.yticks(np.linspace(prange[i][0], prange[i][1], tb[i]))
				plt.ylabel(Name[i], size = 25)
				ax.yaxis.set_major_locator(MaxNLocator(nbins = nbins[i], prune="upper"))
		if i<j:
			plt.axis('on')
			plt.xticks([])
			plt.yticks([])
			if i==n_dims - 1:
				plt.xticks(np.linspace(prange[j][0], prange[j][1], tb[j]))
				ax.xaxis.set_major_locator(MaxNLocator(nbins = nbins[j], prune="upper"))
			if j==0 and i != j:
				plt.yticks(np.linspace(prange[i][0], prange[i][1], tb[i]))
				ax.yaxis.set_major_locator(MaxNLocator(nbins = nbins[i], prune="upper"))

		# plot two distribution this is a diagnoal plot
		if i == j:
			plt.xlim(prange[j][0], prange[i][1])
			A1, B1, C1 = plt.hist(flat_p1[i],  100, range=[prange[j][0], prange[i][1]], color='b', histtype = 'step', normed = True, linewidth = 1.0)
			A2, B2, C2 = plt.hist(flat_p2[i],  100, range=[prange[j][0], prange[i][1]], color='r', histtype = 'step', normed = True, linewidth = 1.0)
			plt.ylim(0, np.max([A1, A2]))
		# plot dataset 1 if this is an upper triangle plot
		if i > j:
			plt.xlim(prange[j][0], prange[j][1])
			plt.ylim(prange[i][0], prange[i][1])
			H, x, y = np.histogram2d(flat_p1[j], flat_p1[i], bins=50, range=np.array([(prange[j][0], prange[j][1]), (prange[i][0], prange[i][1])]), normed = True)
			plt.imshow(np.flipud(H.T), cmap="Blues" , extent = [prange[j][0], prange[j][1], prange[i][0], prange[i][1]], aspect='auto')
		# plot dataset 2 if this is an lower triangle plot
		if i < j:
			plt.xlim(prange[j][0], prange[j][1])
			plt.ylim(prange[i][0], prange[i][1])
			H, x, y = np.histogram2d(flat_p2[j], flat_p2[i], bins=50, range=np.array([(prange[j][0], prange[j][1]), (prange[i][0], prange[i][1])]), normed = True)
			plt.imshow(np.flipud(H.T), cmap="Reds" , extent = [prange[j][0], prange[j][1], prange[i][0], prange[i][1]], aspect='auto')



plt.subplots_adjust(wspace=0., hspace=0., top = 0.99, bottom = 0.08, left = 0.08, right = 0.99)

plt.savefig("corner.png", dpi = 100)
plt.show()











