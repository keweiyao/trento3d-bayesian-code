"""
Singular value decomposition of bare(rescaled) data
"""

if __name__ == "__main__":
	print __doc__

import numpy as np
import matplotlib.pyplot as plt


class data_converter:
	__slots__ = ('origin_data_', 'N_truncate_', 's_value_', 'V_matrix_', 'n_samples_', 'Z_', 'norm_weight_')
	def __init__(self, origin_data, N_truncate):
		self.origin_data_ = origin_data
		self.n_samples_ = np.shape(origin_data)[0]
		U, self.s_value_, self.V_matrix_ = np.linalg.svd(self.origin_data_, full_matrices=True)
		print "Shape of U, s, V: ", U.shape, self.s_value_.shape, self.V_matrix_.shape
		self.Z_ = np.sqrt(self.n_samples_)*np.dot(self.origin_data_, self.V_matrix_.T)
		self.N_truncate_ = N_truncate
		self.norm_weight_ = np.array(self.s_value_**2)
		self.norm_weight_ /= np.sum(self.norm_weight_)
	def get_new_obs(self):		
		temp = self.Z_.T[0:self.N_truncate_]
		mean = np.mean(temp, axis = 1)
		maxabs =  np.max(np.abs(temp), axis = 1)
		return temp, mean, maxabs
	def convert_column_data(self, exp_data):
		if np.shape(exp_data) != (self.V_matrix_.shape[0],):
			print "data dimension error"
		else:
			return np.sqrt(self.n_samples_)*np.dot(exp_data, self.V_matrix_.T)

	def data_reconstruct(self,weights):
		if np.shape(weights) != (self.N_truncate_,):
			print "weights dimension error"
		else:
			return 1.0/np.sqrt(self.n_samples_)*np.dot(weights, self.V_matrix_[0:self.N_truncate_,:])
	def plot_weight(self):
		sum_w = [ np.sum(self.norm_weight_[0:i]) for i in range(1, len(self.norm_weight_)) ]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(np.arange(1, len(sum_w)+1), sum_w,'ro-')
		ax.axis([0, 10, sum_w[0]*0.9, 1.1])
		f = open("weight.txt", 'w')
		for s in sum_w:
			f.write("%s "%s)
		plt.xlabel("n", size=15)
		plt.ylabel(r"$\sum_{i=1}^n w_i$", size = 15)
		plt.show()


