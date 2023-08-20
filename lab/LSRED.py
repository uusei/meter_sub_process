import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy.linalg as sl
import time


def fast_lts(xi,yi,r=0.6,n=3):
    zi=[]
    ni=[]
    ei=[]
    ZO=np.zeros((3,3))
    lamba=np.array([[0, 0, 2],[0, -1, 0],[2, 0, 0]],np.float32)
    S1=np.vstack((np.hstack((lamba,ZO)),np.hstack((ZO,ZO))))

    SA=shuffle_array(int(len(xi)*r))
    xia=xi[SA]
    yia=yi[SA]

    zi=np.vstack((xia.T*xia.T,xia.T*yia.T,yia.T*yia.T,xia.T,yia.T,np.ones((len(xia),1)).T))
    Z1=np.dot(zi,zi.T)

    
    for i in range(0,n):
        aa = characteristic_value(Z1,S1)
        ef1=error_cal(zi,aa)
        SA=shuffle_array(int(len(xi)*r))
        xia=xi[SA]
        yia=yi[SA]
        zi=np.vstack((xia.T*xia.T,xia.T*yia.T,yia.T*yia.T,xia.T,yia.T,np.ones((len(xia),1)).T))
        Z1=np.dot(zi,zi.T)
        aa = characteristic_value(Z1,S1)
        ef2 = error_cal(zi,aa)
        if ef1<ef1:
           continue
        else:
            ni.append(aa)
            ei.append(ef2)
            SA=shuffle_array(int(len(xi)*r))
            xia=xi[SA]
            yia=yi[SA]

            zi=np.vstack((xia.T*xia.T,xia.T*yia.T,yia.T*yia.T,xia.T,yia.T,np.ones((len(xia),1)).T))
            Z1=np.dot(zi,zi.T)
    ni=np.array(ni)
    vei=np.argmin(abs(np.array(ei)))
    vni=ni[vei,:]
    print(vni)
    return ni,ei

def characteristic_value(zi,si):
    (eva,evt)=sl.eig(zi,si)
    VV=np.argmin(abs(eva))
    aa=evt[:,VV]
    return aa

def shuffle_array(len):
    ran = np.arange(len)
    np.random.shuffle(ran)
    return ran

def error_cal(zi,aa):
    epslion=abs(np.dot(zi.T,aa))
    EF=sum(epslion)
    return EF

def make_num():
    xi=[]
    yi=[]
    for i in range(0,30):
        thr=3.14*(i-0.5)/30
        xi.append(10*mt.cos(thr))
        yi.append(2*mt.sin(thr))
    return np.array(xi),np.array(yi)