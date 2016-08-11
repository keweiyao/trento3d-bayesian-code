"""
Gaussian process manager: use george
"""

if __name__ == "__main__":
	print __doc__

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import george

class gp_manager:
	__slots__ = ('model_design_', 'model_obs_', 'model_err_', 'n_dims_', 'gp_', 'kern_', 'pf_')
	def __init__(self, model_design, model_obs, model_err):
		print ">>>>>>>>>>>>>new Gaussian process for a principle component<<<<<<<<<<<<<<<<<"
		self.model_design_ = model_design
		self.model_obs_ = model_obs
		self.model_err_ = model_err
		self.n_dims_ = np.shape(self.model_design_)[1]
		ptemp = np.exp(np.array(np.random.uniform(-4,0,self.n_dims_+2)))
		self.kern_ = ptemp[0]*george.kernels.ExpSquaredKernel(ptemp[1:self.n_dims_+1], ndim=self.n_dims_) + george.kernels.WhiteKernel(ptemp[self.n_dims_+1], ndim=self.n_dims_)
		
		self.gp_ = george.GP(self.kern_, mean=np.mean(self.model_obs_))
		self.gp_.compute(self.model_design_)

		# -------Optimize gaussian process by maximizing the likelihood----------------
		par, results = self.gp_.optimize(self.model_design_, self.model_obs_)
		self.gp_.kernel[:] = par
		current_lln = self.gp_.lnlikelihood(self.model_obs_)
		par_final = par
		for i in range(30):
			self.gp_.kernel[:] = np.exp(np.random.uniform(-4,0,self.n_dims_+2))
			par, results = self.gp_.optimize(self.model_design_, self.model_obs_)
			self.gp_.kernel[:] = par
			new_lln = self.gp_.lnlikelihood(self.model_obs_)
		if new_lln > current_lln:
			current_lln = new_lln
			par_final = par
		print par_final
		self.gp_.kernel[:] = par_final
		print "final log likelihood: ", self.gp_.lnlikelihood(self.model_obs_)

	def predict(self, pinput):
		mean, cov = self.gp_.predict(self.model_obs_, pinput)
		return mean




