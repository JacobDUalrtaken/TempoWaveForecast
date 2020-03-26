# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:49:44 2020

@author: Okan D Lab1
"""

import sys
sys.path.append('C:/Users/Okan D Lab1/.spyder-py3/2020/time series')

import numpy as np
import matplotlib.pyplot as plt
import ewtpy
import my_vmdpy
from PyEMD import EMD
import pywt
import torch
from sklearn import metrics

plt.close('all')
T = 1000
t = np.arange(1,T+1)/T

f1 = 5*t + 2*np.power(t,2)

f2 = np.sin(2*np.pi*100*np.append(t[0:500],np.zeros(500)))
f3 = 2*np.sin(2*np.pi*200*np.append(np.zeros(500),t[500:1000]))

# f2 = np.cos(2*np.pi*100*t)
# f3 =  2*np.sin(2*np.pi*200*t)


f = f1 +f2 + f3 +np.random.randn(t.size) * (0.1)




plt.figure( )
plt.subplot(411)
plt.plot(f)
plt.subplot(412)
plt.plot(f1)
plt.subplot(413)
plt.plot(f2)
plt.subplot(414)
plt.plot(f3)









'''============================================================Wavelet Transform==========================='''
wavelet = 'dmey'
cA2,cD2,cD1 = pywt.wavedec(f, wavelet, mode='periodic', level= 2, axis=0)


coeff_A2 = cA2,                     np.zeros(cD2.size),     np.zeros(cD1.size)
coeff_D2 = np.zeros(cA2.size),      cD2,                    np.zeros(cD1.size)
coeff_D1 = np.zeros(cA2.size),      np.zeros(cD2.size),     cD1


A2 = pywt.waverec(coeff_A2, wavelet, mode='periodic', axis=0)
D2 = pywt.waverec(coeff_D2, wavelet, mode='periodic', axis=0)
D1 = pywt.waverec(coeff_D1, wavelet, mode='periodic', axis=0)

recon_f = A2+D2+D1

erorr =  metrics.mean_squared_error(f,recon_f)

plt.figure( )
plt.subplot(411)
plt.plot(f)

plt.subplot(412)
plt.plot(A2)
plt.title('A2')

plt.subplot(413)
plt.plot(D2)
plt.title('D2')
plt.subplot(414)
plt.plot(D1)
plt.title('D1')


plt.suptitle('DWT')
plt.show()






'''==============================================================EMD======================================='''
emd = EMD()
IMFs = emd(f)

plt.figure( )
plt.subplot(611)
plt.plot(f)
plt.subplot(612)
plt.plot(IMFs[0,:])
plt.subplot(613)
plt.plot(IMFs[1,:])
plt.subplot(614)
plt.plot(IMFs[2,:])
plt.subplot(615)
plt.plot(IMFs[3,:])
plt.subplot(616)
plt.plot(IMFs[4,:])



plt.suptitle('EMD')
plt.show()








'''============================================================== VMD ====================================='''

#. some sample parameters for VMD  
alpha = 1000     # moderate bandwidth constraint  
tau = 0.01          #  time-step of the dual ascent ( pick 0 for noise-slack ) 
K = 3              #  modes  
DC = 1             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-6          # tolerance of convergence criterion; typically around 1e-6


#. Run actual VMD code  
u, u_hat, omega = my_vmdpy.VMD(f, alpha, tau, K, DC, init, tol)  
 #    u       - the collection of decomposed modes
 #   u_hat   - spectra of the modes
 #   omega   - estimated mode center-frequencies
    
plt.figure( )
plt.subplot(411)
plt.plot(f)
plt.subplot(412)
plt.plot(u[0,:])
plt.subplot(413)
plt.plot(u[1,:])
plt.subplot(414)
plt.plot(u[2,:])
plt.suptitle('VMD')
plt.show()

'''==============================================================EWT ====================================='''

ewt,  mfb ,boundaries = ewtpy.EWT1D(f, N = 4)    

plt.figure( )
plt.subplot(411)
plt.plot(f)
plt.subplot(412)
plt.plot(ewt[:,0])
plt.subplot(413)
plt.plot(ewt[:,1])
plt.subplot(414)
plt.plot(ewt[:,2])
plt.suptitle('EWT')

ff = np.fft.fft(f)
freq=2*np.pi*np.arange(0,len(ff))/len(ff)

Fs = 1
if Fs !=-1:
    freq=freq*Fs/(2*np.pi)
    boundariesPLT=boundaries*Fs/(2*np.pi)
else:
    boundariesPLT = boundaries

ff = abs(ff[:ff.size//2])#one-sided magnitude
freq = freq[:freq.size//2]


# plt.figure(3)
# plt.plot(freq,ff)
# for bb in boundariesPLT:
#     plt.plot([bb,bb],[0,max(ff)],'r--')
# plt.title('Spectrum partitioning')
# plt.xlabel('Hz')
# plt.show()

