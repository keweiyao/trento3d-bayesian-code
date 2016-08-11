import numpy as np
import matplotlib.pyplot as plt
import h5py

fout = h5py.File("exp-data.hdf5", 'w')

#----------------p+Pb, ATLAS--------------------------
f = np.loadtxt("./raw-data/pPb-atlas.dat").T
eta = f[0]
dNdy = []
err = []
cL = [0, 1, 5, 10, 20, 30, 40, 60]
cH = [1, 5, 10, 20, 30, 40, 60, 90]
for c in range(8):
	dNdy.append(f[3+c*5])
	err.append(f[4+c*5]+f[6+c*5])
dNdy = dNdy[::-1]
err = err[::-1]
gp = fout.create_group("pPb")
gp.create_dataset("eta", data = eta)
for c in range(8):
	gpc = gp.create_group("cen-%d"%c)
	gpc.attrs.create("cen-cut", data = np.array([cL[c], cH[c]]))
	gpc.create_dataset("dNdy", data = dNdy[c])
	gpc.create_dataset("err", data = err[c])


#---------------Pb+Pb, ALICE-------------------------
dNdy = []
err = []
cL = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
cH = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
#--------------data-1, 0-5, 5-10, 10-20, 20-30----------
f = np.loadtxt("./raw-data/PbPb-alice-high.dat").T
eta = f[0][:-2]
for c in range(4):
	dndy = f[3+c*3][:-2]
	er = f[4+c*3][:-2]
	dNdy.append(dndy)
	err.append(er)
	plt.errorbar(eta, dNdy[-1], [err[-1], err[-1]])
#--------------data-2, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90----------------
f = np.loadtxt("./raw-data/PbPb-alice-low.dat").T
for c in range(6):
	dndy = f[3+c*5]
	er = f[4+c*5]+f[6+c*5]
	# flip and paste data
	dndy = np.concatenate((dndy[-1:-7:-1], dndy) )
	er = np.concatenate((er[-1:-7:-1], er) )
	dNdy.append(dndy)
	err.append(er)
	plt.errorbar(eta, dNdy[-1], [err[-1], err[-1]])

plt.show()


gp = fout.create_group("PbPb")
gp.create_dataset("eta", data = eta)
for c in range(10):
	gpc = gp.create_group("cen-%d"%c)
	gpc.attrs.create("cen-cut", data = np.array([cL[c], cH[c]]))
	gpc.create_dataset("dNdy", data = dNdy[c])
	gpc.create_dataset("err", data = err[c])

fout.close()



