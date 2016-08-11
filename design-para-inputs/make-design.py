import sys
sys.path.append('../')
import mtd.params as params
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

# model
model = int(sys.argv[1])
# parameter file
fout = h5py.File("parameter-%d.hdf5"%model, 'w')

# dimension of parameter space
names = ["flut", "norm-pPb", "norm-PbPb", "mean", "std", "skew", "Jacobi"]
n_dims = 7
# number of model samples
n_samples = 300

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

# >>>----------Latin Hypercube Sampling-----------------------------------
p = params.params(n_dims, n_samples)
p.set_range(prange)
p.generate_design()
p_bare = p.bare_design_
p_phy = p.phy_design_
for i in range(n_samples):
	outfilestring_pPb = """
projectile = p
projectile = Pb
number-events = 10000
quiet = true
output = pPb%d.hdf5
reduced-thickness = 0.0
fluctuation = %f
nucleon-width = 0.45
cross-section = 7.0
normalization = %f
rapidity-mean = %f
rapidity-width = %f
rapidity-skew = %f
rapidity-kurtosis = 0.0
skew-schemes = %d
<pt2/mt2> = %f
grid-max = 5
grid-step = 0.1
eta-max = 12.0
eta-step = 64
switch-3d = 1
out-3d = 0
"""%(i, p_phy[i][0], p_phy[i][1], p_phy[i][3], p_phy[i][4], p_phy[i][5], model, p_phy[i][6])

	outfilestring_PbPb = """
projectile = Pb
projectile = Pb
number-events = 10000
quiet = true
output = PbPb%d.hdf5
reduced-thickness = 0.0
fluctuation = %f
nucleon-width = 0.45
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
"""%(i, p_phy[i][0], p_phy[i][1]/p_phy[i][2], p_phy[i][3], p_phy[i][4], p_phy[i][5], model, p_phy[i][6])
	myFile = open("./input-%d/input-pPb-%d.txt"%(model, i), "w")
	myFile.write(outfilestring_pPb)
	myFile.close()
	myFile = open("./input-%d/input-PbPb-%d.txt"%(model, i), "w")
	myFile.write(outfilestring_PbPb)
	myFile.close()
# <<<----------LHS-----------------------------------
fout.create_dataset("design-bare", data=p_bare)
fout.create_dataset("design-phy", data=p_phy)
fout.close()
