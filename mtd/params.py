"""
Latin hypercube generator: use pyDOE
"""

if __name__ == "__main__":
	print __doc__

import numpy as np
import matplotlib.pyplot as plt
import pyDOE

def rescale(x, prange):
	x = np.array(x)
	prange = np.array(prange)
	return np.array([x[:,i]*(prange[i,1]-prange[i,0]) + prange[i,0] for i in range(len(prange))]).T

def anti_rescale(x, prange):
	x = np.array(x)
	prange = np.array(prange)
	return np.array([(x[:,i]-prange[i,0])/(prange[i,1]-prange[i,0]) for i in range(len(prange))]).T



class params:
	__slots__ = ['n_dims_', 'n_samples_', 'option_','prange_', 'bare_design_', 'phy_design_']
	def __init__(self, n_dims=1, n_samples=10, option='maximin'):
		self.n_dims_ = n_dims
		self.n_samples_ = n_samples
		self.option_ = option
		self.prange_ = np.array([np.array([0,1])]*n_dims)
	
	def set_range(self, lims):
		if np.shape(lims) != (self.n_dims_, 2):
			print "dimension mismatch"
			return
		else:
			self.prange_ = np.array(lims)
			
	def generate_design(self):
		self.bare_design_ = pyDOE.lhs(self.n_dims_, samples=self.n_samples_, criterion=self.option_)		
		self.phy_design_ = rescale(self.bare_design_, self.prange_)


	def plot_design(self, option='bare'):
		if option == 'bare':
			data = self.bare_design_
			plot_range = np.array([np.array([0,1])]*self.n_dims_)
		elif option == 'phy':
			data = self.phy_design_
			plot_range = self.prange_
		fig = plt.figure()
		for i in range(self.n_dims_):
			for j in range(0,i+1):
				ax= fig.add_subplot(self.n_dims_, self.n_dims_, (i)*self.n_dims_ + j+1)
				if i==j:
					ax.set_xlim(plot_range[i])
   					ax.hist(data[:,i], normed=True)
				else:
					ax.set_xlim(plot_range[i])
					ax.set_ylim(plot_range[j])
					ax.scatter(data[:,i], data[:,j])
		plt.show()


