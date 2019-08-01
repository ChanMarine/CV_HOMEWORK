import cv2
import numpy as np
import math
from matplotlib import pyplot as plt 


def calTriArea(x):#通过行列式计算面积
    epsilon = 10
    cal_x = np.hstack((x,np.ones((3,1))))
    det_x = abs(np.linalg.det(cal_x)/2)
    if det_x < epsilon:
        return False
    else:
        return True

#用于检查随机选取的四个点对是否满足非共线性要求(不能有三点共线)
def goCheckPoints(x):
    #通过三点围成面积是否为0判断三点是否共线
    all_idxs = np.arange(x.shape[0])
    if calTriArea(x[all_idxs!=0,:]) and calTriArea(x[all_idxs!=1,:]) and calTriArea(x[all_idxs!=2,:]) and calTriArea(x[all_idxs!=3,:]):
        return True
    else:
        return False    

#用于从数据点集中随机选择出四对点
def getRandom4Pairs2dPoint(A,B):
    all_idxs = np.arange(A.shape[0])  # 获取下标索引
    while True:
        np.random.shuffle(all_idxs)  # 打乱下标索引
        a = A[all_idxs[0:4],:]
        b = B[all_idxs[0:4],:]
        if goCheckPoints(a):
            return all_idxs[0:4],all_idxs[4:]

def calSqareErrors(a,b,model):
    a = np.hstack((a,np.ones((a.shape[0],1))))
    tempHomo = np.dot(a,model)
    x = tempHomo[:,2].reshape(tempHomo.shape[0],1)
    temp = tempHomo/x
    error = np.sum((b-temp[:,0:2])**2,axis=1)
    return error    

#用于从数据点集中随机选择出四对点
def getRandom4Pairs2dPoint(A,B):
    all_idxs = np.arange(A.shape[0])  # 获取下标索引
    while True:
        np.random.shuffle(all_idxs)  # 打乱下标索引
        a = A[all_idxs[0:4],:]
        b = B[all_idxs[0:4],:]
        if goCheckPoints(a):
            return all_idxs[0:4],all_idxs[4:]

#更新最大迭代次数
def updateMaxIters(p,m,r):#p置信度,m样本集总数,r本次迭代样本估计样本内点数目
    num = math.log(1-p)
    w = r/m
    deno = math.log(1-w**4)
    maxIters = round(num/deno)
    return maxIters   

#计算单应性矩阵,主要通过特征分解的方式
def getLsHomography(a,b):
    alen = a.shape[0]
    A = np.zeros((9,9))
    for i in range(alen):
        l1 = np.array([a[i,0],a[i,1],1,0,0,0,-a[i,0]*b[i,0],-a[i,1]*b[i,0],-b[i,0]])
        l2 = np.array([0,0,0,a[i,0],a[i,1],1,-a[i,0]*b[i,1],-a[i,1]*b[i,1],-b[i,1]])
        l1 = l1.reshape(1,9)
        l2 = l2.reshape(1,9)
        A = A + np.dot(l1.T,l1)+np.dot(l2.T,l2)
    retval,_,e_vecs = cv2.eigen(A)
    if retval:
        H = e_vecs[-1,:]
        H = H.reshape(3,3)
        H = H/H[2,2]
    return H

def ransanc(A,B,maxiters=2000,p=0.99,th=3):#A,B为成对的匹配点(A.i与B.i匹配),maxiters最大迭代次数,p模型置信度,th判断是否内点的阈值
    plen = A.shape[0]#总的数据点数目
    iters = 1
    bestMask = np.zeros(plen)#内点数目最多时,对应的掩模
    maxInnerPointNums = 0 #最大的内点数目
    while iters <= maxiters:
        inner_idx,test_idx = getRandom4Pairs2dPoint(A,B)#随机抽取4个点对,作为内点
        model = getLsHomography(A[inner_idx,:],B[inner_idx,:])#由4个点对估计模型参数
#         model,_ = cv2.findHomography(A[inner_idx,:],B[inner_idx,:])
        testerror = calSqareErrors(A[test_idx,:],B[test_idx,:],model.T)
        also_idx = test_idx[testerror<th]
        innerPointsNum = 4+len(also_idx)
        if innerPointsNum > maxInnerPointNums:
            maxInnerPointNums = innerPointsNum
            bestMask[inner_idx] = 1
            bestMask[also_idx] = 1
        maxiters = updateMaxIters(p,plen,innerPointsNum)
        iters+=1 
    innerA = A[bestMask>0,:]
    innerB = B[bestMask>0,:]
#     plt.scatter(A[bestMask>0,0],A[bestMask>0,1],c ='r')
#     plt.scatter(A[bestMask==0,0],A[bestMask==0,1],c ='g')
    finalmodel = getLsHomography(innerA,innerB)
#     finalmodel,_ = cv2.findHomography(innerA,innerB)
    return finalmodel.T,bestMask

                                                        



