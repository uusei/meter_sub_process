import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy.linalg as sl
import time

def least_square(xi,yi):
    zi=[]
    ZO=np.zeros((3,3))
    lamba=np.array([[0, 0, 2],[0, -1, 0],[2, 0, 0]],np.float32)
    S1=np.vstack((np.hstack((lamba,ZO)),np.hstack((ZO,ZO))))

    zi=np.vstack((xi.T*xi.T,xi.T*yi.T,yi.T*yi.T,xi.T,yi.T,np.ones((len(xi),1)).T))
    Z1=np.dot(zi,zi.T)
    aa=characteristic_value(Z1,S1)
    return aa

def characteristic_value(zi,si):
    (eva,evt)=sl.eig(zi,si)
    VV=np.argmin(abs(eva))
    aa=evt[:,VV]
    return aa

def make_num():
    xi=[]
    yi=[]
    for i in range(0,30):
        thr=3.14*(i-0.5)/30
        xi.append(10*mt.cos(thr))
        yi.append(2*mt.sin(thr))
    return np.array(xi),np.array(yi)

t1 = time.time()
xi,yi = make_num()
al1 = least_square(xi,yi)
print(al1)
t2 = time.time()
print(int(round((t2-t1) * 1000)))
