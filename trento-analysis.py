import numpy as np
import matplotlib.pyplot as plt
import mtd.params as params
import mtd.gp_manager as gp_manager
import mtd.data_SVD as data_SVD
import emcee
import sys
import os
import h5py


# model
model = int(sys.argv[1])
N_truncate_pPb = int(sys.argv[2])
N_truncate_PbPb = int(sys.argv[3])
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

#---------------------------------------------------------
#					MTD/Bayesian analysis
#---------------------------------------------------------


#---------------------------------------------------------
#					Import deisgned parameter sets
#---------------------------------------------------------
fpara = h5py.File("design-para-inputs/parameter-%d.hdf5"%model, 'r')
para_dataset = fpara['design-bare'].value
p_norm = []
fpara.close()
#---------------------------------------------------------
#                   Import experimental data
#---------------------------------------------------------
fexp = h5py.File('exp-data/exp-data.hdf5', 'r')
#-----------pPb data----------------------------
eta_pPb = fexp['pPb']['eta'].value
dndy_pPb = []
err_pPb = []
for c in range(8):
	dndy_pPb.append( fexp['pPb']['cen-%d'%c]['dNdy'].value )
	err_pPb.append( fexp['pPb']['cen-%d'%c]['err'].value )
dndy_pPb = np.array(dndy_pPb).flatten()
err_pPb = np.array(err_pPb).flatten()
#-----------PbPb data----------------------------
eta_PbPb = fexp['PbPb']['eta'].value
dndy_PbPb = []
err_PbPb = []
for c in range(10):
	dndy_PbPb.append( fexp['PbPb']['cen-%d'%c]['dNdy'].value )
	err_PbPb.append( fexp['PbPb']['cen-%d'%c]['err'].value )
dndy_PbPb = np.array(dndy_PbPb).flatten()
err_PbPb = np.array(err_PbPb).flatten()

fexp.close()

#---------------------------------------------------------
#                   Import model calculation
#---------------------------------------------------------
outputdir = "phy-model-output/model-%d/"%model
obs_pPb = []
obs_PbPb = []
for i in range(n_samples):
	filename_pPb = outputdir + "pPb-obs-%d.txt"%i
	filename_PbPb = outputdir + "PbPb-obs-%d.txt"%i
	if os.path.isfile(filename_pPb) and os.path.isfile(filename_PbPb):
		calc_pPb = np.loadtxt(filename_pPb)
		calc_PbPb = np.loadtxt(filename_PbPb)
		if np.max(calc_pPb) > 120.:
			continue
		obs_pPb.append(calc_pPb)
		obs_PbPb.append(calc_PbPb)
		p_norm.append(para_dataset[i])
obs_pPb = np.array(obs_pPb)
obs_PbPb = np.array(obs_PbPb)
p_norm = np.array(p_norm)

# >>>----------Prior Plot-------------------------------------------
# center the data by dndy
scale_pPb = np.ones_like(dndy_pPb)*50.
center_pPb = dndy_pPb
obs_pPb_norm = (obs_pPb - center_pPb)/scale_pPb

scale_PbPb = np.ones_like(dndy_PbPb)*1600.
center_PbPb = dndy_PbPb
obs_PbPb_norm = (obs_PbPb - center_PbPb)/scale_PbPb
# plot prior
for r in obs_pPb_norm:
	plt.plot( r, 'b-', linewidth = 0.2, alpha = 1.0, label="Prior")
plt.xlabel("index")
plt.ylabel("dN/deta")
plt.show()
for r in obs_PbPb_norm:
	plt.plot( r, 'b-', linewidth = 0.2, alpha = 1.0, label="Prior")
plt.xlabel("index")
plt.ylabel("dN/deta")
plt.show()
# <<<----------Prior Plot-------------------------------------------

# >>>----------Data SVD-------------------------------------------
converter_pPb = data_SVD.data_converter(obs_pPb_norm, N_truncate_pPb)
Z_pPb_obs, mean_pPb, maxabs_pPb = converter_pPb.get_new_obs()
Z_pPb_err = converter_pPb.convert_column_data(err_pPb/scale_pPb)
for i in range(len(Z_pPb_obs)):
	Z_pPb_obs[i] = (Z_pPb_obs[i]-mean_pPb[i])/maxabs_pPb[i] # rescale
	Z_pPb_err[i] = Z_pPb_err[i]/maxabs_pPb[i] # rescale
converter_pPb.plot_weight()

converter_PbPb = data_SVD.data_converter(obs_PbPb_norm, N_truncate_PbPb)
Z_PbPb_obs, mean_PbPb, maxabs_PbPb = converter_PbPb.get_new_obs()
Z_PbPb_err = converter_PbPb.convert_column_data(err_PbPb/scale_PbPb)
for i in range(len(Z_PbPb_obs)):
	Z_PbPb_obs[i] = (Z_PbPb_obs[i]-mean_PbPb[i])/maxabs_PbPb[i] # rescale
	Z_PbPb_err[i] = Z_PbPb_err[i]/maxabs_PbPb[i] # rescale
converter_PbPb.plot_weight()
# <<<----------Data SVD-------------------------------------------

# >>>----------GP training-------------------------------------------
gp_pPb = []
for i in range(N_truncate_pPb):
	gptemp = gp_manager.gp_manager(p_norm, Z_pPb_obs[i], 0.0)
	gp_pPb.append(gptemp)

gp_PbPb = []
for i in range(N_truncate_PbPb):
	gptemp = gp_manager.gp_manager(p_norm, Z_PbPb_obs[i], 0.0)
	gp_PbPb.append(gptemp)
# <<<----------GP training-------------------------------------------

# >>>------------MCMC-----------------------------------

def lnprob(pv_):
	if (pv_ >= 0).all() and (pv_ < 1.0).all():
		w = pv_[0]
		pvc = pv_[1::]
		Z_pPb_predict = np.array([ gp_pPb[i].predict(np.array([pvc]))[0]  for i in range(N_truncate_pPb) ])
		x_pPb = Z_pPb_predict*maxabs_pPb + mean_pPb
		x_pPb = converter_pPb.data_reconstruct(x_pPb)
		x_pPb = x_pPb*scale_pPb + center_pPb

		Z_PbPb_predict = np.array([ gp_PbPb[i].predict(np.array([pvc]))[0]  for i in range(N_truncate_PbPb) ])
		x_PbPb = Z_PbPb_predict*maxabs_PbPb + mean_PbPb
		x_PbPb = converter_PbPb.data_reconstruct(x_PbPb)
		x_PbPb = x_PbPb*scale_PbPb + center_PbPb

		return -(1.-w)*np.sum(((x_pPb - dndy_pPb)/err_pPb)**2) - w*np.sum(((x_PbPb - dndy_PbPb)/err_PbPb)**2) + 200.*np.log(w) + 216.*np.log(1.-w)
	else:
		return -np.inf

n_walkers = 500
p0 = np.random.rand((n_dims+1) * n_walkers).reshape((n_walkers, n_dims+1))
sampler = emcee.EnsembleSampler(n_walkers, n_dims+1, lnprob)

pos, prob, state = sampler.run_mcmc(p0, 500)
sampler.reset()
print "MCMC burn-in finished"
sampler.run_mcmc(pos, 2000)
flat_p = sampler.flatchain


fout_pset = h5py.File("./result/pset-%d.hdf5"%model, 'w')
print flat_p.shape
fout_pset.create_dataset("pset", data = flat_p)
fout_pset.close()

# <<<------------MCMC-----------------------------------

# >>>----------------Prior/Posterior plot----------------------
f_pri_post = h5py.File("./result/priori-post-%d.hdf5"%model, 'w')
ds = []
for i in range(500):
	pset = flat_p[np.random.randint(0, 50000)][1::]
	post_pPb = np.array([gp.predict(np.array([pset]))[0] for gp in gp_pPb])
	x_pPb = post_pPb*maxabs_pPb + mean_pPb
	x_pPb = converter_pPb.data_reconstruct(x_pPb)
	x_pPb = x_pPb*scale_pPb + center_pPb
	
	post_PbPb = np.array([gp.predict(np.array([pset]))[0] for gp in gp_PbPb])
	x_PbPb = post_PbPb*maxabs_PbPb + mean_PbPb
	x_PbPb = converter_PbPb.data_reconstruct(x_PbPb)
	x_PbPb = x_PbPb*scale_PbPb + center_PbPb

	ds.append(np.concatenate((x_pPb, x_PbPb)))
ds = np.array(ds)
f_pri_post.create_dataset("post", data = ds)
f_pri_post.close()
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
# <<<----------------Corner Plot-------------------------------










