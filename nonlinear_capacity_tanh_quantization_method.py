# -*- coding: utf-8 -*-
"""Nonlinear_capacity_tanh.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_k31cO_5W3LFcAmwA1eWRXWXuK9V5PBv
"""

import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import numpy.random as rnd
import matplotlib.pyplot as plt

def Gauss(N,max_sig):
  x = np.linspace(-max_sig,max_sig,N)
  delta = x[1]-x[0]
  x_new = np.append(x - delta/2, x[-1] + delta/2)
  q_x_new = 0.5 + 0.5*special.erf(x_new/np.sqrt(2))
  q_x_new[0] = 0
  q_x_new[-1] = 1
  f_x = np.ediff1d(q_x_new)
  return f_x

def entropy_y(varx,varz,Nu,max_sig):
  eps = 1e-20

  varu = varx + varz
  delta_u = (2*max_sig/Nu)*np.sqrt(varu)

  u_pts = np.linspace(-max_sig*np.sqrt(varu),max_sig*np.sqrt(varu),Nu)

  f_u = Gauss(int(Nu),max_sig)

  u_bnds = np.append(u_pts - delta_u/2, u_pts[-1] + delta_u/2)
  y_bnds = np.tanh(u_bnds)

  delta_y = np.ediff1d(y_bnds) + eps

  h_y = np.sum(-f_u*np.log(f_u/delta_y))

  return h_y

def entropy_y_given_x(varx,varz,Nx,Nz,max_sig):
  eps = 1e-20
  f_x = Gauss(Nx,max_sig)
  f_x = np.reshape(f_x,[1,Nx])

  f_z = Gauss(int(Nz),max_sig)
  f_z = np.reshape(f_z,[Nz,1])

  x_pts = np.linspace(-max_sig*np.sqrt(varx),max_sig*np.sqrt(varx),Nx)
  x_pts = np.reshape(x_pts,[1,Nx])

  z_pts = np.linspace(-max_sig*np.sqrt(varz),max_sig*np.sqrt(varz),Nz)
  delta_z = (2*max_sig/Nz)*np.sqrt(varz)
  z_bnds = np.append(z_pts - delta_z/2, z_pts[-1] + delta_z/2)
  z_bnds = np.reshape(z_bnds,[Nz+1,1])

  u_bnd_gx = z_bnds + x_pts
  y_bnd_gx = np.tanh(u_bnd_gx)

  delta_y = np.diff(y_bnd_gx,axis=0) + eps
  h_y_Gx = np.dot(f_x,np.sum(-f_z*np.log(f_z/delta_y),axis=0))

  return h_y_Gx

SNR_list = [x*2.0 for x in range(11)] # I don't know why it's a list

Nx = int(1e3)
Nz = int(1e3)
Nu = int(1e3)

max_sig = 4.0

varz = 1

Gaussian_ach = [] #why are these lists ????
AWGN_capacity = []

for snr in SNR_list:
  varx = 10**(snr/10)
  h_y = entropy_y(varx,varz,Nu,max_sig)
  h_y_Gx = entropy_y_given_x(varx,varz,Nx,Nz,max_sig)

  Gaussian_ach.append(h_y-h_y_Gx)
  AWGN_capacity.append(0.5*np.log(1+varx/varz))

plt.plot(SNR_list,AWGN_capacity,label='AWGN Capacity',marker = 'x')
plt.plot(SNR_list,Gaussian_ach,label='Gaussian_ach  '+ r'$\left(\frac{x}{(1+x^4)^{1/4}}\right)$',marker = 'o')
plt.legend(loc='upper left')
plt.xlim([0.0,20.0])
plt.xlabel('SNR (dB)')
plt.ylabel('Rate (nats)')
plt.grid(True)
plt.show()