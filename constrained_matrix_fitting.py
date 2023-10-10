# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:58:31 2023

@author: User
"""

import numpy as np
from math import pi
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydicom
from skimage.restoration import unwrap_phase
import cv2
LF = 2*pi*42.576e6*3



def read_image(files):
    
    ds = pydicom.dcmread(files)
    d = ds.pixel_array*ds.RescaleSlope + ds.RescaleIntercept
    
    return d

def amptitude (idx,te):
    
   
    phase_shift =  np.array([75.3,-63.9,-249,-314,-342,-396,-434,-485])*1j*2*pi
    sigma = phase_shift*te
    
    if idx == 0:
        a = 1
    if idx == 1:
        a = np.exp(sigma[0]) + 4*np.exp(sigma[1]) -1.43*np.exp(sigma[2]) + 6*np.exp(sigma[3]) + 2.87 * np.exp(sigma[4]) + 6*np.exp(sigma[5]) + 72.5*np.exp(sigma[6])+9*np.exp(sigma[7])
        
    if idx == 2:
        a = 2*np.exp(sigma[0]) + 0.896*np.exp(sigma[2]) + 2.21*np.exp(sigma[4]) -4.84*np.exp(sigma[6])
        
 
        
    return a 


def peak_sum(P,i):
    ndb = P[2]/(P[1]+1e-9)
    #ndb = ndb.real
    
 
    
    phase_shift =  np.array([75.3,-63.9,-249,-314,-342,-396,-434,485])*1j*2*pi
    alpha = [2*ndb+1,4,0.896*ndb-1.43,6,2.21*ndb+2.87,6,72.5-4.84*ndb,9]
    peak_s = 0
    
    for p in range(0,phase_shift.shape[0]):
        peak_s+=(alpha[p]*np.exp(phase_shift[p]*TE[i]))
        
    return peak_s


def field_matrix(field,TE):
    
    fdm = (0+0j)*np.ones((14,14))
    phase = field*TE
    for i in range(14):
        fdm[i,i] = np.exp(phase[i])
        
    return fdm

def B_matrix(P):
    TE = np.array([1.201e-3,1.909e-3,2.618e-3,3.327e-3,4.035e-3,4.744e-3,5.452e-3,6.161e-3,6.87e-3,7.578e-3,8.287e-3,8.995e-3,9.704e-3,10.413e-3 ])
    B = (0+0j)*np.ones((14,3))
    
    #B[:,1] = 1
    
    for i in range(B.shape[0]):
        for j in range(0,B.shape[1]):
            if j == 0:
                B[i,j] =  (P[0]+P[1]*peak_sum(P,i) )*TE[i]
            if j >=1:
                B[i,j] = amptitude(j-1, TE[i])
            
            
    return B

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def circle_computing(p_map):
    
    radius1 = 8
    xc1,yc1 = 43,75
   
    h,w = p_map.shape[0],p_map.shape[1]

    mask1 = create_circular_mask(h,w,[xc1, yc1],radius1)
    
    radius2 = 8
    xc2,yc2 = 73,75
    h,w = p_map.shape[0],p_map.shape[1]
    mask2 = create_circular_mask(h,w,[xc2, yc2],radius2)
    
   
    mask1 = mask1.astype('int')
    mask2 = mask2.astype('int')
    
    masked_map_1 = p_map*mask1
    masked_map_2 = p_map*mask2
    
    mean_1 = np.sum(masked_map_1)/np.sum(mask1)
    mean_2 = np.sum(masked_map_2)/np.sum(mask2)
    
    return mean_1,mean_2,mask1,mask2
    
    
    

if __name__ =='__main__':
    
    n_echo = 14
    C = 3
    cut_step = 2
    
    #ref_ff = (ref['F']/(ref['F']+ref['W']+1e-9))[200:240,65:125]
    data = loadmat('phantom_1.mat')
    slice_idx = 3
    image = data['imDataParams']['images'][0][0]
    slicing_x_l,slicing_x_r = 110,220
    slicing_y_l,slicing_y_r = 110,220
    test_slice = image[slicing_x_l:slicing_x_r, slicing_y_l:slicing_y_r,slice_idx,0,:]
    
    #test_slice = image[:,:,slice_idx,0,:]
    water = np.ones_like(test_slice[:,:,0])
    fat = np.ones_like(test_slice[:,:,0])
    Ff = np.ones_like(test_slice[:,:,0])
    ndb_map = np.ones_like(test_slice[:,:,0])
    cl_map = np.ones_like(test_slice[:,:,0])
    nmidb_map = np.ones_like(test_slice[:,:,0])
    phase_map = np.ones_like(test_slice[:,:,0])
    TE = np.array([1.201e-3,1.909e-3,2.618e-3,3.327e-3,4.035e-3,4.744e-3,5.452e-3,6.161e-3,6.87e-3,7.578e-3,8.287e-3,8.995e-3,9.704e-3,10.413e-3 ])
  
    for x in tqdm(range(test_slice.shape[0])):
        for y in range(test_slice.shape[1]): 
                   
                real_data = test_slice[x,y,:]
                S = np.reshape(real_data,(14,1))
                
                A = (1+1j)*np.ones((14,3))
                for r in range(n_echo):
                    for c in range(C):
                        A[r,c] = amptitude(c, TE[r])
                        
                A = np.matrix(A)   
                AT = A.H
                phi = 0
                fdm_inv = field_matrix(-phi, TE)
                
                P = np.linalg.inv(AT*A+1e-9 )*AT *(fdm_inv)*(S)
                for num_iter in range(50):
                    
                    B = B_matrix(P)
                    B = np.matrix(B)
                    BT = B.H
            
                    fmp = field_matrix(-phi, TE)
                   
                    Delta = np.linalg.inv( (BT*B+1e-9) ) *BT * (  fmp.dot(S) - A*P)
                    Delta = np.array(Delta)
                    phi = phi+Delta[0]
                   
                    fdm_inv = field_matrix(-phi, TE)
                    
                    P = np.linalg.inv(AT*A+1e-9 )*AT *(fdm_inv)*(S)
                    if np.sqrt(Delta[0][0].real**2+Delta[0][0].imag**2) < 1e-3:
                        break
                P = np.array(P)
                phase_map[x,y] = P[0]
                ndb = P[2]/(P[1]+1e-9)
                #nmidb = 0.093*ndb**2
                #cl = 16.8+0.25*ndb
                water[x,y] = P[0]
               
                Ff[x,y] = P[1]
                ndb_map[x,y] = ndb
                
             
              
        
  
    
    w = water  
    nmidb_map = 0.448*ndb_map-0.714
    cl_map = 0.378*ndb_map+16.3
    f_final = Ff*(99.94+0.266*ndb)
    water_final = w
    ff = f_final/(water_final+f_final+1e-9)
    
    plt.figure(1)
    plt.imshow(np.abs(ff.real),vmin = 0,vmax=0.15,cmap='jet')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(image[slicing_x_l:slicing_x_r,slicing_y_l:slicing_y_r,slice_idx,0,0].real**2+image[slicing_x_l:slicing_x_r, slicing_y_l:slicing_y_r,slice_idx,0,0].imag**2,cmap='gray')
   
    plt.figure(3)
    plt.imshow(np.abs(ndb_map.real),vmin = 1,vmax=6,cmap='inferno')
    plt.colorbar()
    
    fraction_1,fraction_2,mask1,mask2 = circle_computing(np.abs(ff.real))
    ndb_1,ndb_2,mask1,mask2 = circle_computing(np.abs(ndb_map.real))
    plt.imshow(mask1+mask2,alpha=0.4,cmap='gray')
    print(fraction_1,fraction_2)
    print(ndb_1,ndb_2)
    
    
   
    
    
    
            
    
    