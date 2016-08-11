from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from scipy.special import lpn



def re_normalize(c12, cut):
	Nx, Ny = c12.shape
	nx = np.linspace(-cut, cut, Nx+1)/cut
	ny = np.linspace(-cut, cut, Ny+1)/cut
	ndx = nx[1]-nx[0]
	ndy = ny[1]-ny[0]
	e1 = np.sum(c12, axis = 1)*ndx/2.
	e2 = np.sum(c12, axis = 0)*ndy/2.
	d11 = np.outer(e1, e2)
	nc12 = c12/d11
	return nc12

def calc_Tmn(cut, m, n, Nx, Ny):
	Nx, Ny = c12.shape
	binx = np.linspace(-cut, cut, Nx+1)/cut
	biny = np.linspace(-cut, cut, Ny+1)/cut
	nx = 0.5*(binx[0:-1] + binx[1:])
	ny = 0.5*(biny[0:-1] + biny[1:])
	ndx = nx[1]-nx[0]
	ndy = ny[1]-ny[0]
	Tm = [lpn(m, x)[0][-1] for x in nx]
	Tn = [lpn(n, x)[0][-1] for x in nx]
	Tmn = 0.5*np.sqrt((m+0.5)*(n+0.5))*(np.outer(Tm, Tn) + np.outer(Tn, Tm))
	return Tmn, ndx, ndy

def calc_a11(c12, cut, m, n):
	Nx, Ny = c12.shape
	Tmn, ndx, ndy = calc_Tmn(cut, m, n, Nx, Ny)
	amn = np.sum(c12*Tmn)*ndx*ndy
	return amn

filename = sys.argv[1]
f = h5py.File(filename)
c12 = f['c12'].value

# re-normalize
nc12 = re_normalize(c12, 2.4)

fig = plt.figure(figsize=plt.figaspect(0.6)*1.5)
ax = fig.gca(projection='3d')
y = np.linspace(-2.4, 2.4, 11)
y1, y2 = np.meshgrid(y, y)
surf = ax.plot_surface(y1, y2, nc12-1, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

a11 = calc_a11(nc12-1., 2.4, 1, 1)
a22 = calc_a11(nc12-1., 2.4, 2, 2)
a33 = calc_a11(nc12-1., 2.4, 3, 3)
a44 = calc_a11(nc12-1., 2.4, 4, 4)

a12 = calc_a11(nc12-1., 2.4, 1, 2)
a13 = calc_a11(nc12-1., 2.4, 1, 3)
a14 = calc_a11(nc12-1., 2.4, 1, 4)

a23 = calc_a11(nc12-1., 2.4, 2, 3)
a24 = calc_a11(nc12-1., 2.4, 2, 4)
print a11**0.5, a22**0.5, a33**0.5, a44**0.5
#print (-a12)**0.5, (-a13)**0.5, (-a14)**0.5
#print (-a23)**0.5, (-a24)**0.5
plt.show()

