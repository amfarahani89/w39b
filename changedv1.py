#!/home/moon/kmolaverdi/pRT/pRT/bin/python
# -*- coding: utf-8 -*-
"""
By Karan Molaverdikhani    
"""

# import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# import emcee as emcee
import rebin as rb
import time

# import pickle

# from schwimmbad import MPIPool

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

import pymultinest

import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')

# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fontsize=12


import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
matplotlib.rcParams['axes.formatter.useoffset'] = False

# plt.rcParams["font.family"] = "Times New Roman"
params = {'mathtext.default': 'regular','font.size': 12 }          
plt.rcParams.update(params)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def minmaxerr(x,xerr):
    i_x_min=np.argmin(x)
    x_min=x[i_x_min]-xerr[i_x_min]
    i_x_max=np.argmax(x)
    x_max=x[i_x_max]+xerr[i_x_max]
    return x_min,x_max

def offset_reset(x1,xerr1,y1,yerr1,x2,xerr2,y2,yerr2):
    x_min1,x_max1 = minmaxerr(x1,xerr1)
    x_min2,x_max2 = minmaxerr(x2,xerr2)
    x_min_shared = max(x_min1,x_min2)    
    x_max_shared = min(x_max1,x_max2)
    
    indx_shared1=np.where((x1>=x_min_shared)&(x1<=x_max_shared))
    median1=np.median(y1[indx_shared1])
    indx_shared2=np.where((x2>=x_min_shared)&(x2<=x_max_shared))
    median2=np.median(y2[indx_shared2])
    
    offset = median1-median2 #how much move dataset 2 with respect to dataset 1
    
    return offset

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 - loads the spectra of the planet
prefix='baseline_'
# prefix='baseline_most_fast_1p0to6p0micron_'

output_dir = 'out/'


path='/home/moon/kmolaverdi/pRT/retrieval/W39b_NS_systematic/data/'
# path='/Users/karan/LMU/pRT/retrieval/W39b_NS_systematic/data/'

file_LBT = path+'w39_prism_firefly_trans_spec_20July2022_v1.txt'
data = np.genfromtxt(file_LBT,skip_header=1)
wl_obs_LBT,r_ratio_obs_LBT, dwl_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

x1 = wl_obs_LBT
xerr1 = dwl_obs_LBT/2.
y1 = r_ratio_obs_LBT
yerr1 = dr_ratio_obs_LBT
# yerr1=abs(2.*r_ratio_obs_LBT*yerr1)
# x=x1
# xerr=xerr1
# y=y1
# yerr=yerr1

file_LBT = path+'W39_HST_Fischer2016.txt'
data = np.genfromtxt(file_LBT,skip_header=1)
wl_obs_LBT,dwl_obs_LBT, r_ratio_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

x2 = wl_obs_LBT
xerr2 = dwl_obs_LBT#/2.
y2 = r_ratio_obs_LBT
yerr2 = dr_ratio_obs_LBT

offset1=1.70544308e-04

# offset1 = offset_reset(x1,xerr1,y1,yerr1,x2,xerr2,y2,yerr2)



file_LBT = path+'W39_Spitzer_Fischer2016.txt'
data = np.genfromtxt(file_LBT,skip_header=1)
wl_obs_LBT,dwl_obs_LBT, r_ratio_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

x3 = wl_obs_LBT
xerr3 = dwl_obs_LBT#/2.
y3 = r_ratio_obs_LBT
yerr3 = dr_ratio_obs_LBT

offset2=2.33660385e-04

# offset2 = offset_reset(x1,xerr1,y1,yerr1,x3,xerr3,y3,yerr3)



file_LBT = path+'W39_HST_Wakeford2018.txt'
data = np.genfromtxt(file_LBT,skip_header=1)
wl_obs_LBT,dwl_obs_LBT, r_ratio_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

x4 = wl_obs_LBT
xerr4 = dwl_obs_LBT/2.
y4 = r_ratio_obs_LBT
yerr4 = dr_ratio_obs_LBT

offset3=-5.07809485e-05

# # offset3 = offset_reset(x1,xerr1,y1,yerr1,x4,xerr4,y4,yerr4)

# # x=np.concatenate((x,x4))
# # xerr=np.concatenate((xerr,xerr4))
# # y=np.concatenate((y,y4+offset3))
# # yerr=np.concatenate((yerr,yerr4))



file_LBT = path+'W39_FORS2_Nikolov2016.txt'
data = np.genfromtxt(file_LBT,skip_header=1)
wl_obs_LBT,dwl_obs_LBT, r_ratio_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

x5 = wl_obs_LBT
xerr5 = dwl_obs_LBT#/2.
y5 = r_ratio_obs_LBT
yerr5 = dr_ratio_obs_LBT

offset4=1.31269673e-05

# offset4 = offset_reset(x1,xerr1,y1,yerr1,x5,xerr5,y5,yerr5)





# file_LBT = path+'W39_ACAM_Kirk2019.txt'
# data = np.genfromtxt(file_LBT,skip_header=1)
# wl_obs_LBT,dwl_obs_LBT, r_ratio_obs_LBT, dr_ratio_obs_LBT  = data[:,0], data[:,1], data[:,2], data[:,3]

# x6 = wl_obs_LBT
# xerr6 = dwl_obs_LBT/2.
# y6 = r_ratio_obs_LBT
# yerr6 = dr_ratio_obs_LBT

# offset5 = offset_reset(x1,xerr1,y1,yerr1,x6,xerr6,y6,yerr6)

# x=np.concatenate((x,x6))
# xerr=np.concatenate((xerr,xerr6))
# y=np.concatenate((y,y6+offset5))
# yerr=np.concatenate((yerr,yerr6))





# drop the observed points out of model wavelenght range       
wln_dn=2.0
wln_up=3.0                                 
indx_wl_coverage = np.where((x1>=wln_dn)&(x1<=wln_up))
x1 = x1[indx_wl_coverage]
xerr1 = xerr1[indx_wl_coverage]
y1 = y1[indx_wl_coverage]
yerr1 = yerr1[indx_wl_coverage]

indx_wl_coverage = np.where((x2>=wln_dn)&(x2<=wln_up))
x2 = x2[indx_wl_coverage]
xerr2 = xerr2[indx_wl_coverage]
y2 = y2[indx_wl_coverage]
yerr2 = yerr2[indx_wl_coverage]

indx_wl_coverage = np.where((x3>=wln_dn)&(x3<=wln_up))
x3 = x3[indx_wl_coverage]
xerr3 = xerr3[indx_wl_coverage]
y3 = y3[indx_wl_coverage]
yerr3 = yerr3[indx_wl_coverage]

indx_wl_coverage = np.where((x4>=wln_dn)&(x4<=wln_up))
x4 = x4[indx_wl_coverage]
xerr4 = xerr4[indx_wl_coverage]
y4 = y4[indx_wl_coverage]
yerr4 = yerr4[indx_wl_coverage]

indx_wl_coverage = np.where((x5>=wln_dn)&(x5<=wln_up))
x5 = x5[indx_wl_coverage]
xerr5 = xerr5[indx_wl_coverage]
y5 = y5[indx_wl_coverage]
yerr5 = yerr5[indx_wl_coverage]



# print(len(x1),len(y1))
# print(len(x2),len(y2))
# print(len(x3),len(y3))
# print(len(x4),len(y4))
# print(len(x5),len(y5))



x=np.concatenate((x1,x2))
xerr=np.concatenate((xerr1,xerr2))
# y=np.concatenate((y1,y2+offset1))
yerr=np.concatenate((yerr1,yerr2))

x=np.concatenate((x,x3))
xerr=np.concatenate((xerr,xerr3))
# y=np.concatenate((y,y3+offset2))
yerr=np.concatenate((yerr,yerr3))

x=np.concatenate((x,x4))
xerr=np.concatenate((xerr,xerr4))
# y=np.concatenate((y,y4))
yerr=np.concatenate((yerr,yerr4))

x=np.concatenate((x,x5))
xerr=np.concatenate((xerr,xerr5))
# y=np.concatenate((y,y5+offset4))
yerr=np.concatenate((yerr,yerr5))




i_x_min=np.argmin(x)
x_min=x[i_x_min]-xerr[i_x_min]
i_x_max=np.argmax(x)
x_max=x[i_x_max]+xerr[i_x_max]

# print(x)
# print(xerr)
# print(y)
# print(yerr)


# # Check the data!
# fig = plt.figure()
# plt.errorbar(x, y, yerr=yerr, fmt=".r")
# print('data_check.pdf')
# plt.savefig('data_check.pdf',bbox_inches=0.)
# plt.close()




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 2 - setup the atmosphere object

# ~~~~~~~~~~ Planetary Pararmeters ~~~~~~~~~~~~
file_out = 'W39b'
#https://docs.google.com/document/d/1zxvMoDXCeZOkr1RQWUFTR5KKf3iXFjpIXnKDzT2qxY0/edit
Rs = 0.932*9.95 # in Jupiter radius #for transit
# # R_pl = 1.138*nc.r_jup_mean
gravity = 2479.*(0.281)*(1.279)**-2. #put in (mass) and (radius)**-2 in Jupiuter unit

P0=0.01

# ~~~~~~~~~~ Atmosphere object Pararmeters ~~~~~~~~~~~~
line_species=['H2O_Exomol','CO2']
rayleigh_species=['H2', 'He']
continuum_opacities=['H2-H2', 'H2-He']
wlen_bords_micron=[x_min, x_max]


# ~~~~~~~~~~ Atmosphere object Construction ~~~~~~~~~~~~
atmosphere = Radtrans(line_species = line_species, \
      rayleigh_species = rayleigh_species, \
      continuum_opacities = continuum_opacities, \
      wlen_bords_micron = wlen_bords_micron)

log_pressure_bords=[-6,1]
pressures = np.logspace(log_pressure_bords[0], log_pressure_bords[1], 20) #<<<<<<<<<<<< increase nz for final run <<<<<<<<<
atmosphere.setup_opa_structure(pressures)





#Number of free parameters
ndim = len(line_species)+1+1+1+0 #species(n), Rp(1 param), temperature profile(1 param), haze/cloud(1 param), 0 offsets precalculated

n_params=ndim


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 0 - Some functions, including the Likelihood function

def calc_MMW(abundances):
    MMWs = {}
    MMWs['H2'] = 2.
    MMWs['He'] = 4.
    MMWs['H2O'] = 18.
    MMWs['CH4'] = 16.
    MMWs['CO2'] = 44.
    MMWs['CO'] = 28.
    MMWs['Na'] = 23.
    MMWs['K'] = 39.
    MMWs['Li'] = 7.
    MMWs['NH3'] = 17.
    MMWs['HCN'] = 27.
    MMWs['C2H2'] = 26.
    MMWs['PH3'] = 34.
    MMWs['H2S'] = 34.
    MMWs['VO'] = 67.
    MMWs['TiO'] = 64.
    MMWs['OH'] = 17.
    MMWs['CaH'] = 41.
    MMWs['CaO'] = 56.
    MMWs['SiO'] = 44.
    MMWs['SO2'] = 64.
    MMWs['SO3'] = 32.+3*16
    MMWs['FeH'] = 57.
    MMWs['BeH'] = 10.
    MMWs['C2H2'] = 26.
    MMWs['NS'] = 14.+32.
    MMWs['O3'] = 3.*16
    MMWs['H3'] = 3.
    MMWs['CrH'] = 52.+1.
    MMWs['Ti'] = 48.
    MMWs['TiH'] = 48.+1.
    MMWs['NaH'] = 23.+1.
    MMWs['C2H4'] = 2*12.+4.
    MMWs['Al'] = 27.
    MMWs['AlH'] = 27.+1.
    MMWs['AlO'] = 27.+16.
    MMWs['CH3'] = 12.+3.
    MMWs['Ca'] = 40.
    MMWs['Fe'] = 56.
    MMWs['CS'] = 12.+32    
    MMWs['H2CO'] = 2+12+16.
    MMWs['H2O2'] = 2+2*16.
    MMWs['H3O+'] = 3+16.
    MMWs['Mg'] = 24.
    MMWs['Mg+'] = 24.
    MMWs['MgH'] = 24.+1
    MMWs['MgO'] = 24.+16
    MMWs['NO'] = 14.+16
    MMWs['NaOH'] = 23.+16+1
    MMWs['O'] = 16
    MMWs['O+'] = 16
    MMWs['O2'] = 2*16
    MMWs['PO'] = 31+16
    MMWs['SH'] = 32+1
    MMWs['Si'] = 28
    MMWs['Si+'] = 28
    MMWs['SiH2'] = 28+2
    MMWs['SiO2'] = 28+2*16
    MMWs['Ti+'] = 48
    MMWs['V'] = 51
    MMWs['V+'] = 51

    

    MMW = 0.
    for key in abundances.keys():
        if key == 'CO_all_iso_Chubb':
            MMW += abundances[key]/MMWs['CO']
        elif key == 'H2O_HITEMP':
            MMW += abundances[key]/MMWs['H2O']
        elif key == 'H2O_Exomol':
            MMW += abundances[key]/MMWs['H2O']
        elif key == 'HDO':
            MMW += abundances[key]/MMWs['H2O']
        elif key == 'Na_allard':
            MMW += abundances[key]/MMWs['Na']
        elif key == 'K_allard':
            MMW += abundances[key]/MMWs['K']
        elif key == 'TiO_all_Exomol':
            MMW += abundances[key]/MMWs['TiO']
        elif key == 'SO2_Chubb':
            MMW += abundances[key]/MMWs['SO2']
        elif key == 'SO3_Chubb':
            MMW += abundances[key]/MMWs['SO3']
        elif key == 'CH4_Chubb':
            MMW += abundances[key]/MMWs['CH4']
        elif key == 'HDO_Chubb':
            MMW += abundances[key]/MMWs['H2O']
        elif key == 'NS_Chubb':
            MMW += abundances[key]/MMWs['NS']
        elif key == 'H3+':
            MMW += abundances[key]/MMWs['H3']
        elif key == 'Al+':
            MMW += abundances[key]/MMWs['Al']
        elif key == 'Ca+':
            MMW += abundances[key]/MMWs['Ca']
        elif key == 'Fe+':
            MMW += abundances[key]/MMWs['Fe']
        elif key == 'CS_Chubb':
            MMW += abundances[key]/MMWs['CS']
        else:
            MMW += abundances[key]/MMWs[key]
    
    
    return 1./MMW
    
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ret_model(atmosphere,gravity,P0,params_i):
    abundances = {}    # Abundances log(abund)=[-10,0]
    log_abund_non_H2=0.
    for i in range(len(atmosphere.line_species)):
        log_abund_non_H2+=10.**params_i[i]
        abundances[atmosphere.line_species[i]] = 10.**params_i[i] * np.ones_like(atmosphere.press)

    log_abund_H2He = 1. - log_abund_non_H2

    abundances['H2'] = np.ones_like(atmosphere.press)*log_abund_H2He*0.766
    abundances['He'] = np.ones_like(atmosphere.press)*log_abund_H2He*0.234
        
    R_pl     =      params_i[len(atmosphere.line_species)]*nc.r_jup_mean
    # kappa_IR = 10.**params_i[len(atmosphere.line_species)+1]
    # gamma    =      params_i[len(atmosphere.line_species)+2]
    # T_int    =      params_i[len(atmosphere.line_species)+3]
    # T_equ    =      params_i[len(atmosphere.line_species)+4]
    T_equ    =      params_i[len(atmosphere.line_species)+1]
    # kappa_zero=10.**params_i[len(atmosphere.line_species)+2]
    # gamma_scat=     params_i[len(atmosphere.line_species)+3]
    # Pcloud   = 10.**params_i[len(atmosphere.line_species)+4]
    Pcloud   = 10.**params_i[len(atmosphere.line_species)+2]

    # temperature = nc.guillot_global(atmosphere.press*1e-6, kappa_IR,gamma,gravity,T_int,T_equ)
    temperature = np.zeros_like(atmosphere.press)+T_equ
    MMW = calc_MMW(abundances)
    atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                    # kappa_zero = kappa_zero, gamma_scat = gamma_scat,Pcloud=Pcloud)
                    Pcloud=Pcloud)
                    # )
        
    return nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate likelihood function
def lnlike(cube, ndim, nparams):
    # print("H$_2$O: %7.3f, Rp: %7.3f, Teq: %7.3f, kappa0: %9.5f, gamma: %9.5f, Pc: %9.5f" % (cube[0],cube[1],cube[2],cube[3],cube[4],cube[5]))
    # ret model
    # start_time = time.time()

    wln,model=ret_model(atmosphere, gravity, P0, cube) # model in R_jup
    # print("--- ret_model %s seconds ---" % (time.time() - start_time))

    if np.sum(np.isnan(model))>0:  
        return -np.inf

    model=(model/Rs)**2.
    
    # Modeled flux in observed bin
    # Modeled flux in observed bin
    flux_rb1, dflux_rb1,_,_,_ = rb.rebin_2sides(x1, xerr1, wln, model)    
    flux_rb2, dflux_rb2,_,_,_ = rb.rebin_2sides(x2, xerr2, wln, model)    
    flux_rb3, dflux_rb3,_,_,_ = rb.rebin_2sides(x3, xerr3, wln, model)    
    flux_rb4, dflux_rb4,_,_,_ = rb.rebin_2sides(x4, xerr4, wln, model)    
    flux_rb5, dflux_rb5,_,_,_ = rb.rebin_2sides(x5, xerr5, wln, model)    
    flux_rb=np.concatenate((flux_rb1,flux_rb2))
    flux_rb=np.concatenate((flux_rb,flux_rb3))
    flux_rb=np.concatenate((flux_rb,flux_rb4))
    flux_rb=np.concatenate((flux_rb,flux_rb5))
    dflux_rb=np.concatenate((dflux_rb1,dflux_rb2))
    dflux_rb=np.concatenate((dflux_rb,dflux_rb3))
    dflux_rb=np.concatenate((dflux_rb,dflux_rb4))
    dflux_rb=np.concatenate((dflux_rb,dflux_rb5))
    # print("--- rebin_2sides %s seconds ---" % (time.time() - start_time))

    # offset1   = cube[len(atmosphere.line_species)+3]
    # offset2   = cube[len(atmosphere.line_species)+4]
    # offset3   = cube[len(atmosphere.line_species)+5]
    # offset4   = cube[len(atmosphere.line_species)+6]
    # offset5   = cube[len(atmosphere.line_species)+12]
    y_tot=np.concatenate((y1,y2+offset1))
    y_tot=np.concatenate((y_tot,y3+offset2))
    y_tot=np.concatenate((y_tot,y4+offset3))
    y_tot=np.concatenate((y_tot,y5+offset4))
    # y_tot=np.concatenate((y_tot,y6+offset5))

    inv_sigma2 = 1.0/(yerr**2.0+dflux_rb**2.0)
    ll_tot = -0.5*(np.sum((y_tot-flux_rb)**2*inv_sigma2 - np.log(inv_sigma2)))
    # print(ll_tot)
    return ll_tot


def normalize(value,val_min,val_max):
    return (value-val_min)/(val_max-val_min)

def denormalize(value,val_min,val_max):
    return value*(val_max-val_min)+val_min


def prior(cube, ndim, nparams):
    # log_abundances
    for i in range(len(line_species)):
    	low = -12
    	up = -1
    	cube[i]=denormalize(cube[i], low, up) # uniform prior between -12:-1
   
    # R_pl
    low=0.8
    up =1.6
    cube[len(line_species)] = denormalize(cube[len(line_species)],low,up)

    
    # # log_kappa_IR = 0.01 #[-4,0]
    # low=-6
    # up =-2
    # cube[len(line_species)+1] = denormalize(cube[len(line_species)+1],low,up)
    # # gamma = 0.4 #[0.1,.9]
    # low=0.2
    # up =0.6
    # cube[len(line_species)+2] = denormalize(cube[len(line_species)+2],low,up)
    # # T_int = 200. #[100,2000]
    # low=100
    # up =600
    # cube[len(line_species)+3] = denormalize(cube[len(line_species)+3],low,up)
    # T_equ = 1200. #[400,4000]
    low=500
    up =1500
    cube[len(line_species)+1] = denormalize(cube[len(line_species)+1],low,up)

    
    # # Haze & Cloud
    # # log_kappa_zero = 0.01 #[-4,0]
    # low=-3
    # up =4
    # cube[len(line_species)+2] = denormalize(cube[len(line_species)+2],low,up)
    # # gamma_scat = -14. #[-30,10]
    # low=-20
    # up =0
    # cube[len(line_species)+3] = denormalize(cube[len(line_species)+3],low,up)
    # # Pcloud = 0.01 #as in pressures: log(Pcloud)=[-5,1]
    low=-6
    up =1
    cube[len(line_species)+2] = denormalize(cube[len(line_species)+2],low,up)
 
    # # offsets
    # low=-0.005
    # up =+0.005
    # cube[len(line_species)+3] = denormalize(cube[len(line_species)+3],low,up)
    # low=-0.005
    # up =+0.005
    # cube[len(line_species)+4] = denormalize(cube[len(line_species)+4],low,up)
    # low=-0.005
    # up =+0.005
    # cube[len(line_species)+5] = denormalize(cube[len(line_species)+5],low,up)
    # low=-0.005
    # up =+0.005
    # cube[len(line_species)+6] = denormalize(cube[len(line_species)+6],low,up)
    # low=-0.005
    # up =+0.005
    # cube[len(line_species)+7] = denormalize(cube[len(line_species)+7],low,up)




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3 - setups and runs mcmc
start_time = time.time()

# with MPIPool() as pool:
#     if not pool.is_master():
#         pool.wait()
#         sys.exit(0)

#     # single thread
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(atmosphere,gravity,P0,Rs,lows,highs,x, xerr,y, yerr),pool=pool)
#     #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(rbfs,column_independent,x, xerr,y, yerr))
    
#     # multiple threads: If your log-probability function takes a significant amount of time (> 1 second or so) to compute then using the parallel sampler actually provides significant speed gains.
#     #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(rbfs,column_independent,x, xerr,y, yerr,prior_condition), threads=3)
    
#     # first run “burn-in” steps in your MCMC chain to let the walkers explore the parameter space a bit and 
#     # get settled into the maximum of the density distribution and save the final position of the walkers 
#     state = sampler.run_mcmc(pos, n_steps_burnin)
#     sampler.reset()
    
#     # run the MCMC for N steps
#     sampler.run_mcmc(state, n_steps)


# Close the processes.
# pool.close()


nlive = 100 # number of live points
pymultinest.run(lnlike, prior, n_params, outputfiles_basename=output_dir+'/'+prefix, resume = False, verbose = True,n_live_points=nlive)



# print("Mean acceptance fraction (0.25-0.5 is the best): {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print("--- %s seconds ---" % (time.time() - start_time))

# parameters = ["pos1", "width", "height1"]

# parameters=['log(H2O)','CH4','log(Na)','Rp','Teq','log(kappa0)','gammascat','log(PC)','offset1','offset2']
# parameters=np.concatenate((np.asarray(line_species),np.asarray(['Rp','Teq','log(kappa0)','gammascat','log(PC)']))).tolist()
parameters=np.concatenate((np.asarray(line_species),np.asarray(['Rp','Teq','log(PC)']))).tolist()

import json
json.dump(parameters, open(output_dir+'/'+prefix+'params.json', 'w')) # save parameter names


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Pickle sampler to use later
# with open(file_out+'.pkl', 'wb') as f:
#     pickle.dump(sampler.chain, f)

# print('Sampler is Pickled! Have fun exploring the results!')
# 








#############################################
# Plot the results

a = pymultinest.Analyzer(outputfiles_basename=output_dir+'/'+prefix, n_params = n_params)
posteriors=a.get_equal_weighted_posterior()
# quantiules
params_p16 = np.empty(ndim)
params_p50 = np.empty(ndim)
params_p84 = np.empty(ndim)
# params=sampler_chain.reshape((-1, ndim))
for i in range(ndim):
    params_p16[i] = np.percentile(posteriors[:,i], 16)
    params_p50[i] = np.percentile(posteriors[:,i], 50)
    params_p84[i] = np.percentile(posteriors[:,i], 84)






fig = plt.figure()
fig.set_size_inches(8., 4.)

gs2 = GridSpec(1,10)
gs2.update(left=0.12, right=0.97, bottom=.11, top=.97,hspace=.07,wspace=.07)
ax1=plt.subplot(gs2[0:1,0:10])


ax1.errorbar(x1, y1, xerr=xerr1, yerr=yerr1, label="w39_prism_firefly_trans_spec_20July2022_v1", 
              fmt='o', color='r', mec='crimson',ecolor='crimson', elinewidth=.5, capsize=0,zorder=2,ms=3)

ax1.errorbar(x2, y2+offset1, xerr=xerr2, yerr=yerr2, label="W39_HST_Fischer2016", 
              fmt='o', color='royalblue', mec='b',ecolor='b', elinewidth=.5, capsize=0,zorder=2,ms=3)

ax1.errorbar(x3, y3+offset2, xerr=xerr3, yerr=yerr3, label="W39_Spitzer_Fischer2016", 
              fmt='o', color='seagreen', mec='g',ecolor='g', elinewidth=.5, capsize=0,zorder=2,ms=3)

ax1.errorbar(x4, y4+offset3, xerr=xerr4, yerr=yerr4, label="W39_HST_Wakeford2018", 
              fmt='o', color='orange', mec='darkorange',ecolor='darkorange', elinewidth=.5, capsize=0,zorder=2,ms=3)

ax1.errorbar(x5, y5+offset4, xerr=xerr5, yerr=yerr5, label="W39_FORS2_Nikolov2016", 
              fmt='o', color='gray', mec='k',ecolor='k', elinewidth=.5, capsize=0,zorder=2,ms=3)

# ax1.errorbar(x6, y6+offset5, xerr=xerr6, yerr=yerr6, label="W39_ACAM_Kirk2019", 
#               fmt='o', color='violet', mec='purple',ecolor='purple', elinewidth=.5, capsize=0,zorder=2,ms=3)


ax1.set_xlim(min(x)-max(xerr),max(x)+max(xerr))
# ax1.set_ylim([.0165,.0185])
ax1.set_xlim([.3,6])
# plt.ylim([.016,.0185])
ax1.set_ylabel(r'$(R_p/R_*)^2$')
ax1.set_xscale('log')
#plt.yscale('log')
ax1.set_xlabel('Wavelength ($\mu$m)')

plt.xticks((.4,.5,.6,.7,.8,.9,1,1.2,1.5,2,4,5), ('0.4','0.5','0.6','','0.8','','1.0','1.2','1.5','2','4','5'))


ax1.legend(loc=1, scatterpoints = 1, fontsize=9, ncol=1,numpoints=1)

# fig.tight_layout(rect=[0, 0.02, 1, 0.97])

# plot the distribution of a posteriori possible models
for cube in a.get_equal_weighted_posterior()[::100,:-1]:
    wln,model=ret_model(atmosphere, gravity, P0, cube)
    model=(model/Rs)**2.
    plt.plot(wln,model, '-', color='blue', alpha=0.3, label='data')

print('data_'+file_out+prefix+'.pdf')
plt.savefig('data_'+file_out+prefix+'.pdf',bbox_inches=0.)
plt.close()

import subprocess
subprocess.run(["multinest_marginals.py", output_dir+prefix])
subprocess.run(["multinest_marginals_corner.py", output_dir+prefix])
subprocess.run(["multinest_marginals_fancy.py", output_dir+prefix])
 
