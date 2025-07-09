#!/usr/bin/env python
import os
import sys
import numpy as np
import emcee

def setup_init_chain(n_times=10,nwalkers=20):
    nbunch = 10*n_times
    nstep = 10*n_times
    nwalkers = nwalkers
    return nbunch,nstep,nwalkers

def setup_init_params():
    init_b = 30.
    init_m  = 1.5                                                                                                                              
    return [init_b,init_m]

def getModel(b,m,x):
    return b+m*x

def prepare_data():
    X = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146], dtype = float)
    Y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344], dtype = float)
    sigma_X = np.array([9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5], dtype = float)
    sigma_Y = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22], dtype = float)
    cov = np.diag(sigma_Y[4:]**2)
    cov_inv = np.linalg.inv(cov)
    return X[4:],Y[4:],cov_inv

def lnprior(theta):
    b,m  = theta
    if 0.0 < b < 50. and 0.0 < m < 5.0:
        return 0.0
    return -np.inf

def lnlike(theta,x, y, invcov):
    b,m = theta
    model = getModel(b,m,x)
    diff = y-model
    return -0.5*np.dot(diff, np.dot(invcov, np.transpose(diff)))

def lnprob(theta,x, y, invcov):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lpb = lp + lnlike(theta, x, y, invcov)
    return lpb


dirname = sys.argv[1]
n_times = 20

print("output directory: %s" % dirname)
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

nbunch,nstep,nwalkers = setup_init_chain(n_times=10,nwalkers=20)
x,y,cov_inv = prepare_data()
inits = setup_init_params()

ndim = 2
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x,y, cov_inv))

for i in range(nbunch):
    if i == 0:
        pos = [inits+ 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
    else :
        pos = sampler.chain[:,-1,:]
    sampler.run_mcmc(pos, nstep)
    filename_bunch_chains = os.path.join(dirname, "chains")
    np.save(filename_bunch_chains, sampler.chain)
    filename_bunch_lnprobabilities = os.path.join(dirname, "lnprobabilities")
    np.save(filename_bunch_lnprobabilities, sampler.lnprobability)
    print("%s/%s bunch completed. File written in %s.npy" % (i+1, nbunch, filename_bunch_chains))
