import numpy as np
import h5py
import sys

model = int(sys.argv[1])
# dimension of parameter space
n_dims = 7
# number of model samples
n_samples = 50
# range of parameters
prange = np.zeros([n_dims, 2])
prange[0] = 	[1.0, 5.0]  #fluct
prange[1] = 	[9,    13]	#norm-pPb
prange[2] = 	[1.0,    1.4]	#norm-pPb/norm-PbPb
prange[3] = 	[0,   2.0]	#mean
prange[4] = 	[2.5, 4.5]  #std
if model == 1:
	prange[5] = [0.0, 2.0]  #skew -- model-1
if model == 2:
	prange[5] = [0.0, 0.6]  #skew -- model-2 
prange[6] = 	[0.6, 0.9]  #Jacobi
# p = [fluct, norm, mean, width, skewness, kurtosis, jacobi]
# readin existing posterior samples

f = h5py.File("./result/pset-%d.hdf5"%model, 'r')
ds = f['pset'].value
N_max = len(ds)
for i in range(n_samples):
	index = np.random.randint(0, N_max, 1)[0]
	pst = ds[index][1:]
	ps = np.zeros(n_dims)
	for j in range(n_dims):
		ps[j] = pst[j]*(prange[j][1] - prange[j][0]) + prange[j][0]
	print pst, ps
	outfilestring_PbPb = """
# specify the projectile option twice
projectile = Pb
projectile = Pb
number-events = 10000

# don't print event properties to stdout, save to HDF5
quiet = true
output = PbPb%d.hdf5

reduced-thickness = 0.0

# 1.4 at LHC; > 3.0 at RHIC (The larger this parameter, the smaller the fluctuation)
fluctuation = %f

# 0.43 at LHC; ~0.5 at RHIC
nucleon-width = 0.45

# 3.98 at 200 GeV; 6.4 at 2.76 TeV; 7.0 at 5.02 TeV
cross-section = 6.4

normalization = %f

rapidity-mean = %f
rapidity-width = %f
rapidity-skew = %f
rapidity-kurtosis = 0.0
skew-schemes = %d

<pt2/mt2> = %f

grid-max = 10
grid-step = 0.2
eta-max = 12.0
eta-step = 64
switch-3d = 1
out-3d = 0
"""%(i ,ps[0], ps[1]/ps[2], ps[3], ps[4], ps[5], model, ps[6])

	f = open("./check-input/model-%d/input-PbPb-%d.txt"%(model, i), 'w')
	f.write(outfilestring_PbPb)
	f.close()

