import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys

model = int(sys.argv[1])

fexp = h5py.File('../exp-data/exp-data.hdf5', 'r')
#-----------pPb data----------------------------
eta_pPb = fexp['pPb']['eta'].value

dndy_pPb = []
err_pPb = []
for c in range(8):
	dndy_pPb.append( fexp['pPb']['cen-%d'%c]['dNdy'].value )
	err_pPb.append( fexp['pPb']['cen-%d'%c]['err'].value )
dndy_pPb = np.array(dndy_pPb)
err_pPb = np.array(err_pPb)
#-----------PbPb data----------------------------
eta_PbPb = fexp['PbPb']['eta'].value
dndy_PbPb = []
err_PbPb = []
for c in range(10):
	dndy_PbPb.append( fexp['PbPb']['cen-%d'%c]['dNdy'].value )
	err_PbPb.append( fexp['PbPb']['cen-%d'%c]['err'].value )

fexp.close()
plt.subplot(1,2,1)
for c in range(8):
	plt.plot(eta_pPb, dndy_pPb[c], 'ro-', linewidth = 2.)
plt.subplot(1,2,2)
for c in range(10):
	plt.plot(eta_PbPb, dndy_PbPb[c], 'ro-', linewidth = 2.)

f = h5py.File("priori-post-%d.hdf5"%model, 'r')
ds = f['post'].value.T

ds_pPb = ds[0:432].T
ds_PbPb = ds[432:832].T

plt.subplot(1,2,1)
for i in range(500):
	dndy = ds_pPb[i].reshape(8, 54)
	for c in range(8):
		plt.plot(eta_pPb, dndy[c], 'b-', linewidth = 0.3, alpha=0.3)

plt.subplot(1,2,2)
for i in range(500):
	dndy = ds_PbPb[i].reshape(10, 40)
	for c in range(10):
		plt.plot(eta_PbPb, dndy[c], 'b-', linewidth = 0.3, alpha=0.3)
plt.savefig("post-%d.png"%model, dpi = 100)
plt.show()


