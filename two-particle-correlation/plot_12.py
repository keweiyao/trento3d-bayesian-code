import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os

# interpolate <a1^2> at these naprts
npart = np.linspace(10, 370, 100)

# calculate the mean and std band of rapidity correlation <a1^2> given input folder
def calc_corr(dirname):
	list_a1sqr = []
	for i in range(50):
		fname = dirname+"PbPb-two-corr-%d.dat"%i
		if os.path.isfile(fname):
			ds = np.loadtxt(fname)
			#print fname
			# interpolate between <a1^2> and Npart
			f = interp.interp1d(ds[0], ds[1])
			#calculate at desired points
			a1sqr = f(npart)
			list_a1sqr.append(a1sqr)
	return np.mean(list_a1sqr , axis = 0), np.std(list_a1sqr , axis = 0)	

mean1, std1 = calc_corr("IC-calc/corr-1/")
mean2, std2 = calc_corr("IC-calc/corr-2/")
x1 = [10, 45, 100, 223, 310]
y1 = [0.12658, 0.066719, 0.044044, 0.029586, 0.0225]
x2 = [10, 45, 100, 223, 310]
y2 = [0.10882, 0.0765032, 0.057567, 0.03979, 0.03117]

# ATLAS measurement
exp = np.loadtxt("./data-atlas.dat").T

plt.plot(exp[0], exp[1], 'ko', label = "ATLAS, preliminary")
# ploting +/- std error band
plt.fill_between(npart, mean1-std1, mean1+std1, alpha = 0.5, color = 'b', label = "IC-I")
plt.fill_between(npart, mean2-std2, mean2+std2, alpha = 0.5, color = 'r', label = "IC-II")
plt.plot(x1, y1, 'bo-', markersize = 10, linewidth = 2.0, label = 'IC-I + hybrid')
plt.plot(x2, y2, 'ro-', markersize = 10, linewidth = 2.0, label = 'IC-II + hybrid')




plt.xlabel(r"$N_{part}$", size = 20)
plt.ylabel(r"$\sqrt{\langle a_1^2 \rangle}$", size = 20)
plt.semilogy()
plt.axis([0,400, 1e-2, 5e-1])
plt.legend(loc = 'best', framealpha = 0.0)
plt.show()
