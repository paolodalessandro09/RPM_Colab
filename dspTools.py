#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:35:32 2021

@author: benjamincolburn
"""

import numpy as np
import numpy.linalg as npl
from scipy import signal as sig
from scipy import stats
from scipy import io
#import cvxpy 
import matplotlib.pyplot as plt
import os
#import ITL as itl
#from NLTS import MackeyGlass
import pandas as pd
from sklearn.model_selection import train_test_split
import csv 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from data import lorenz, henon
from scipy.signal import correlate,butter,sosfilt
from scipy.stats import levy_stable
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    import nonlinear_benchmarks
except ImportError:
    nonlinear_benchmarks = None
try:
    import reservoirpy.datasets as timedatasets
except ImportError:
    timedatasets = None

'''
This script houses functions used for signal processing. 
'''

def weightedL2(x,y,w):
    q = x-y
    return np.sqrt((w*q*q).sum())

def shift(x,s):
    '''
    Shift a signal.

    Parameters
    ----------
    x : numpy array
        signal you want to shift.
    s : integer
        Shift amount. 
        positve will shift function right (delayed in time).
        negative will shift function left (shift earlier in time).

    Returns
    -------
    shifted version of x.

    '''
    x1 = np.roll(x,s)
    N = len(x)
    xi = np.arange(0,N) - s
    tf = (xi < 0)
    x1[tf]= 0
    tf = xi > N
    x1[tf] = 0
    
    return x1

def minMaxNorm(x):
    return (x-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0)),np.min(x,axis=0), np.max(x,axis=0)
def aminMaxNorm(x,mini,maxi):
    return (x-mini)/(maxi-mini)

def invMinMaxNorm(x,mini,maxi):
    return (x*(maxi-mini))+mini

def getReg(A,threshold=10, searchStep=1):
    '''
    Finds regularization coefficient so that the condition number is below a certain threshold 

    Parameters
    ----------
    A : numpy Matrix
        matrix you will be regularizing.
    threshold : float
        threshold for acceptable conditioning number

    Returns
    -------
    Regularization Coefficient.

    '''
    L = A.shape[0]
    w,_ = npl.eig(A)
    minE = w.min()
    multiplier = 2
    lam = minE*multiplier
    cNum = npl.cond((A+(lam*np.eye(L))))
    counter = 1
    
    while cNum > threshold:
        lam = multiplier * minE
        multiplier += searchStep
        cNum = npl.cond((A+(lam*np.eye(L))))
        if counter % 100000 == 0:
            x=1
            #print(f'search steps:{counter}, {lam},{cNum}')
        if counter >=1e8:
            x=1
            #print('getReg...Counter Broke')
            break
            
        
        counter += 1
        
    #print(f'Multiplier={multiplier}')
    
    return lam, multiplier


def getInputsJumpNoisedB(Input,Desired,N,L,startidx,E=None,noisedb = 0,pH=0,sdim=False):
    
    t = 0
    if E is None:
        x = np.zeros((N,L))
    else:
        x = np.zeros((N,L//E,E))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    if noisedb != 0 or noisedb != 100:
        nsig = np.sqrt(np.mean(Input**2)*(10**(-noisedb/20)))
        noise  = np.random.normal(0,nsig,len(Input))
        #print(f'noiseStd = {nsig}')
        signalN = Input + noise
    else:
        signalN = Input
    if E is None:
        for i in range(N):
            x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH+startidx]
            t +=1
    else:
        for i in range(N):
            window  =  np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH+startidx]
            count = 0
            t+=1
            for e in range(0,L,E):
                x[i,count,:] = window[e:e+E]
                count+=1
      
    return x,d
    

def createInputs2(systemInfo,seed=0):
    np.random.seed(seed)
    system = systemInfo['System']
    
    if system == 'ParWH0':
        train_val, testset = nonlinear_benchmarks.ParWH()
        
        trainInput = train_val[0].u[0:-2000]
        trainOutput = train_val[0].y[0:-2000]
        valInput = train_val[0].u[-2000:]
        valOutput = train_val[0].y[-2000:]
        testInput = testset[0].u
        testOutput = testset[0].y
        muX = np.mean(trainInput)
        muY = np.mean(trainOutput)
        trainInput -=muX
        trainOutput -= muY
        valInput -=muX
        valOutput -= muY
        testInput -=muX
        testOutput -= muY
        
        #import pdb;pdb.set_trace()
        
        sTrain  = np.random.randint(0,1000)
        if systemInfo['N'] > 25000:
            sTrain = 200
        sVal = np.random.randint(0,1000)
        sTest =   np.random.randint(0,1000)
        sTest =   500
        
        Xtrain,Dtrain =getInputsJumpNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xval,Dval =getInputsJumpNoisedB(valInput,valOutput,1000,systemInfo['L'],sVal+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        return Xtrain,Xval,Xtest,Dtrain,Dval,Dtest
      
        
    if system == 'F160':
        train_val,testset = nonlinear_benchmarks.F16()
        trainInput = train_val[0].u[0:-5000]
        trainOutput = train_val[0].y[0:-5000]
        valInput = train_val[0].u[-5000:]
        valOutput = train_val[0].y[-5000:]
        testInput = testset[0].u
        testOutput = testset[0].y
        sTrain  = np.random.randint(0,15000)
        sVal = np.random.randint(0,1000)
        sTest =   np.random.randint(0,5000)
        
        Xtrain,Dtrain =getInputsJumpNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xval,Dval =getInputsJumpNoisedB(valInput,valOutput,systemInfo['Ntest'],systemInfo['L'],sVal+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
          
        return Xtrain,Xval,Xtest,Dtrain,Dval,Dtest

    if system== 'CO2':
        df = pd.read_csv('/media/benjamin/5CCEC71CCEC6ECF8/KUBUTU/NoTrick/Signals/co2_mm_mlo.csv',comment='#')
        df.columns = ["year", "month", "decimal_date","co2", "co2_interp", "trend","days", "uncertainty"]

        # NOAA convention: missing values are -99.99
        #df.replace(-99.99, np.nan, inplace=True)
    
        # Prefer measured monthly mean if available
        if "average" in df.columns:
            co2_col = "average"
        elif "co2" in df.columns:
            co2_col = "co2"
        else:
            raise ValueError("Could not find CO2 column")
    
        # Drop missing values
        trend = df['trend'].to_numpy()
        df = df.dropna(subset=[co2_col])
        if systemInfo['pH']==0:
            print('WARNING: pH is zero')
        t = df["decimal_date"].to_numpy()
        co2 = df[co2_col].to_numpy()-trend
        trainset = co2[:700]
        valset = co2[700:750]
        testset = co2[750:]
        Xtrain,Dtrain =getInputsJumpNoisedB(trainset,trainset,systemInfo['N'],systemInfo['L'],systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xval,Dval =getInputsJumpNoisedB(valset,valset,50-systemInfo['L']-1,systemInfo['L'],systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testset,testset,systemInfo['Ntest'],systemInfo['L'],systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        Ttrain,_ =getInputsJumpNoisedB(t,t,systemInfo['N'],1,systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Tval,_ =getInputsJumpNoisedB(t,t,50-systemInfo['L']-1,1,systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Ttest,_ =getInputsJumpNoisedB(t,t,systemInfo['Ntest'],1,systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
          
        return Xtrain,Xval,Xtest,Dtrain,Dval,Dtest,Ttrain,Tval,Ttest
    
    if system == 'SARCOS':
        trainset = io.loadmat('/media/benjamin/5CCEC71CCEC6ECF8/KUBUTU/NoTrick/Signals/sarcos_inv.mat')['sarcos_inv'].astype(np.float64)[:,:22]
        valset = trainset[40000:,:22]
        testset = io.loadmat('/media/benjamin/5CCEC71CCEC6ECF8/KUBUTU/NoTrick/Signals/sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float64)[:,:22]
        print(f'train: {trainset.shape}')
        print(f'val: {valset.shape}')
        print(f'test: {testset.shape}')
        import pdb;pdb.set_trace()
        trainIDX  = np.arange(trainset.shape[0])
        valIDX  = np.arange(valset.shape[0])
        testIDX  = np.arange(testset.shape[0])
        np.random.shuffle(trainIDX)
        np.random.shuffle(valIDX)
        np.random.shuffle(testIDX)
        trainidx = np.random.choice(trainIDX,size=systemInfo['N'],replace=False)
        validx = np.random.choice(valIDX,size=systemInfo['Ntest'],replace=False)
        testidx = np.random.choice(testIDX,size=systemInfo['Ntest'],replace=False)
    
        
        
        Xtrain = trainset[trainidx,:-1];Dtrain = trainset[trainidx,-1]
        Xval = valset[validx,:-1];Dval = valset[validx,-1]
        Xtest = testset[testidx,:-1];Dtest = testset[testidx,-1] 

        muX = np.mean(Xtrain,axis=0)
        scaleX = np.std(Xtrain,axis=0)

        Xtrain = (Xtrain-muX)/scaleX
        Xval = (Xval-muX)/scaleX
        Xtest = (Xtest-muX)/scaleX

        muD = np.mean(Dtrain)        
        Dtrain -= muD
        Dval -= muD
        Dtest -= muD
        
          
        return Xtrain,Xval,Xtest,Dtrain,Dval,Dtest
    
    

        
def createInputs(systemInfo,seed=0):
    '''
    Creates Inputs for prediction

    Parameters
    ----------
    systemInfo : Dict
        Dictionary with the system settings stored.
    seed : int, optional
        seed for random number generation. The default is 0.

    Returns
    -------
    Xtrain : np array (N x L)
        Training Inputs.
    Xtest : np array (N x L)
        Testing Inputs.
    dTrain : np array (N x 1)
        Training Desired.
    dTest : np array (N x 1)
        Testing Desired.

    '''
    np.random.seed(seed)
    systemNames = ['MackeyGlass','Lorenz','SS','Noise2Poly','MackeyGlass2','RealEstate','MackeyGlassTest','CO2','SunSpot']
    
    if 'System' not in systemInfo.keys():
        print('System not Specified.')
        return None
    
    system = systemInfo['System']
    
    if system == 'MackeyGlass':
        MK30 = np.squeeze(io.loadmat('./Signals/MK30.mat')['MK30'])
        MK30 = (MK30 - np.mean(MK30))
        trainSet = MK30[1000:4000]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        sTrain = np.random.randint(150,175)
    
        testSet = MK30[4000:]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        sTest =np.random.randint(150,500)
        print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        
        if systemInfo['align'] == True:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],
                                          systemInfo['L'],sTrain,noiseTrain,sdim=systemInfo['sdim'])
            Xtest = Xtrain[systemInfo['N']:]
            dTest = dTrain[systemInfo['N']:]
            Xtrain =Xtrain[:systemInfo['N']]
            dTrain =dTrain[:systemInfo['N']]
        else:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,sdim=systemInfo['sdim'])
            Xtest,dTest = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,sdim=systemInfo['sdim'])
            
    
    if system == 'bops':
        data = np.loadtxt(f'/media/benjamin/5CCEC71CCEC6ECF8/KUBUTU/NoTrick/Keil/Keil2/bop/bops/bop{systemInfo["fileNumber"]}.dat',delimiter=None )
        #print(f'file:{file}')
        xxtemp = data[:,2]
        #idxs = np.where(xtemp==0)
        #xtemp[idxs] = -1
        d = data[:,3]
        dtemp =d/np.max(d)
        diffs = np.abs(xxtemp-dtemp)
        
        #d = d[:,np.newaxis]
        
        x,d = getInputsnoNoise(diffs,dtemp,N=120-(systemInfo['L']+2),L=systemInfo['L'],startidx=systemInfo['L']+1,noisedb = 100,pH=systemInfo['pH'],sdim=False,zeropad=False)
        x2,_ = getInputsnoNoise(xxtemp,dtemp,N=120-(systemInfo['L']+2),L=systemInfo['L'],startidx=systemInfo['L']+1,noisedb = 100,pH=systemInfo['pH'],sdim=False,zeropad=False)
        #x2 = np.round(x2)
        xx = np.zeros((x.shape[0],x.shape[1],2))
        xx[:,:,0] =x;
        for l in range(systemInfo['L']):
            xx[:,l,1] = x2[:,l]
        #xx[:,:,1] = xtemp[-1];
        x =xx
        
        return x[:systemInfo['changePoint'],:,:],x[systemInfo['changePoint']:,:,:],d[:systemInfo['changePoint']],d[systemInfo['changePoint']:]
    
    
    if system == 'MackeyGlassdB':
        MK30 = np.squeeze(io.loadmat('./Signals/MK30.mat')['MK30'])
        MK30 = (MK30 - np.mean(MK30))
        #MK30 += np.random.normal(0,0.)
        trainSet = MK30[1000:4000]
        if systemInfo['AlphaStable']:
            trainSet += levy_stable.rvs(alpha=systemInfo['Stable_Alpha'], beta=0.,loc=0.,scale=systemInfo['Stable_Scale'], size=len(trainSet))
        
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        sTrain = np.random.randint(150,175)
    
        testSet = MK30[4000:]
        sTest =np.random.randint(150,500)
        #print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        Xtrain,dTrain = getPredInputsNoisedB(trainSet,systemInfo['N'],systemInfo['L'],sTrain,systemInfo['wgnVar'],pH=systemInfo['pH'],sdim=systemInfo['sdim'])
        Xtest,dTest = getPredInputsNoisedB(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,systemInfo['wgnVar'],pH =systemInfo['pH'], sdim=systemInfo['sdim'])
        if systemInfo['ImpNoise']:
            dnoise = getImpnoise(len(dTrain),systemInfo['p0'],systemInfo['impmu'])
            dclean = np.copy(dTrain)
            dTrain +=dnoise
            
            return Xtrain,Xtest,dTrain,dTest,dclean
            #dTrain -=np.mean(dTrain)
        
        if systemInfo['AlphaStable']:
            dclean = np.copy(dTrain)
            return Xtrain,Xtest,dTrain,dTest,dclean
        
    if system == 'CO2':
        MK30 = np.squeeze(io.loadmat('./Signals/CO2_data.mat')['MK30'])
        MK30 = (MK30 - np.mean(MK30))
        trainSet = MK30[1000:4000]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        sTrain = np.random.randint(150,175)
    
        testSet = MK30[4000:]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        sTest =np.random.randint(150,500)
        #print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        
        Xtrain,dTrain = getPredInputsNoisedB(trainSet,systemInfo['N'],systemInfo['L'],sTrain,systemInfo['wgnVar'],sdim=systemInfo['sdim'])
        Xtest,dTest = getPredInputsNoisedB(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,systemInfo['wgnVar'],sdim=systemInfo['sdim'])
      
    if system == 'Lorenz':
        Lr = np.squeeze(io.loadmat('./Signals/lorenz.mat')['lorenz2'])
        Lr = (Lr - np.mean(Lr))/np.std(Lr)
        Lr = sig.decimate(Lr,2)
        trainSet = Lr[[np.range(0,2000),np.range(4000,len(Lr))]]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        if seed == 0:
            sTrain = np.random.randint(100,500)
        else:
            sTrain = np.random.randint(300,500)
        
        testSet = Lr[2000:4000]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        if seed == 0:
            sTest =np.random.randint(50,500)
        else:
            sTest =np.random.randint(50,500)
        print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        if systemInfo['align'] == True:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],systemInfo['L'],sTrain,noiseTrain,pH=systemInfo['pH'],sdim=systemInfo['sdim'])
            Xtest = Xtrain[systemInfo['N']:]
            dTest = dTrain[systemInfo['N']:]
            Xtrain =Xtrain[:systemInfo['N']]
            dTrain =dTrain[:systemInfo['N']]
        else:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,pH=systemInfo['pH'],sdim=systemInfo['sdim'])
            Xtest,dTest = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,pH=systemInfo['pH'],sdim=systemInfo['sdim'])
    
    if system =='SS':
        SS = np.load('./Signals/WSS_new.npy')
        trainSet= SS[:10000]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        sTrain = np.random.randint(1500,2000)
        
        
        testSet = SS[10000:20000]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        sTest =np.random.randint(1500,2000)
        
        print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        if systemInfo['align'] == True:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],systemInfo['L'],sTrain,noiseTrain,sdim=systemInfo['sdim'])
            Xtest = Xtrain[systemInfo['N']:]
            dTest = dTrain[systemInfo['N']:]
            Xtrain =Xtrain[:systemInfo['N']]
            dTrain =dTrain[:systemInfo['N']]
        else:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,pH=systemInfo['pH'],sdim=systemInfo['sdim'])
            Xtest,dTest = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,sdim=systemInfo['sdim'])
            
    if system == 'SimpleSS':
        trainSeed = np.random.randint(0,500)
        testSeed = np.random.randint(0,500)
        Xtrain,Dtrain = getSS(systemInfo['N'],systemInfo['L'],1,systemInfo['Powers'],systemInfo['Weights'],seed = trainSeed)
        Xtest,Dtest = getSS(systemInfo['Ntest'],systemInfo['L'],1,systemInfo['Powers'],systemInfo['Weights'],seed = testSeed)
        
        return Xtrain,Xtest,Dtrain,Dtest
    
    if system == 'Silverbox_Validate':
        train_val, _ = nonlinear_benchmarks.Silverbox()
        trainInput = train_val.u[:-20000]
        trainOutput = train_val.y[:-20000]
        testInput = train_val.u[20000:]
        testOutput = train_val.y[20000:]
        sTrain  = np.random.randint(0,10000)
        sTest =   np.random.randint(0,5000)
        
        Xtrain,Dtrain =getInputsJumpNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testInput,testOutput,systemInfo['N'],systemInfo['L'],sTest+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        return Xtrain,Xtest,Dtrain,Dtest
    
    if system == 'ParWH':
        train_val, testset = nonlinear_benchmarks.ParWH()
        trainInput = train_val[0].u[5000:-5000]
        trainOutput = train_val[0].y[5000:-5000]
        testInput = testset[0].u
        testOutput = testset[0].y
        sTrain  = np.random.randint(0,15000)
        sTest =   np.random.randint(0,5000)
        
        Xtrain,Dtrain =getInputsJumpNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        return Xtrain,Xtest,Dtrain,Dtest
    
    if system == 'Laser':
        dataset = timedatasets.santafe_laser()[:,0]
        dataset = (dataset-np.mean(dataset))/np.std(dataset)

        trainset = dataset[1000:8000]
        valset = dataset[0:1000]
        testset = dataset[9000:]
        #import pdb;pdb.set_trace()
        sTrain  = np.random.randint(0,500)
        #sVal  = np.random.randint(0,len(valset)-systemInfo['Ntest']-systemInfo['L'])
        sVal = 0
        #sTest =   np.random.randint(0,len(testset)-systemInfo['Ntest']-systemInfo['L'])
        sTest = 0
        
        Xtrain,Dtrain =getInputsJumpNoisedB(trainset,trainset,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xval,Dval =getInputsJumpNoisedB(valset,valset,systemInfo['Ntest'],systemInfo['L'],sVal+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,Dtest =getInputsJumpNoisedB(testset,testset,systemInfo['Ntest'],systemInfo['L'],sTest+systemInfo['L'],E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        return Xtrain,Xval,Xtest,Dtrain,Dval,Dtest
        
        
    
    if system == 'SimpleSS1':
        trainSeed = np.random.randint(0,500)
        testSeed = np.random.randint(0,500)
        Xtrain,Dtrain = getSS1(systemInfo['N'],systemInfo['L'],2,systemInfo['Powers'],systemInfo['Weights'],seed = trainSeed)
        Xtest,Dtest = getSS1(systemInfo['Ntest'],systemInfo['L'],2,systemInfo['Powers'],systemInfo['Weights'],seed = testSeed)
        
        if systemInfo['wgnVar'] !=100:
            nsig = np.sqrt(np.mean(Dtrain**2)*(10**(-systemInfo['wgnVar']/20)))
            noise  = np.random.normal(0,nsig,len(Dtrain))
        
            Dtrain += noise
        if systemInfo['AlphaStable']:
            Dclean = Dtrain.copy()
            Dtrain += levy_stable.rvs(alpha=systemInfo['Stable_Alpha'], beta=0.,loc=0.,scale=systemInfo['Stable_Scale'], size=len(Dtrain))
            return Xtrain,Xtest,Dtrain,Dtest,Dclean
        #Xtrain  -=np.mean(Xtrain)
        Dtrain -= np.mean(Dtrain)
        
        #Xtest  -=np.mean(Xtest)
        Dtest -= np.mean(Dtest)
        
        
        return Xtrain,Xtest,Dtrain,Dtest
    
    if system == 'SimpleSSE2':
        trainSeed = np.random.randint(0,500)
        testSeed = np.random.randint(0,500)
        Xtrain,Dtrain = getSSE2(systemInfo['N'],systemInfo['L'],np.pi,systemInfo['Powers'],systemInfo['Weights'],seed = trainSeed)
        Xtest,Dtest = getSSE2(systemInfo['Ntest'],systemInfo['L'],np.pi,systemInfo['Powers'],systemInfo['Weights'],seed = testSeed)
        
        Xtrain  -=np.mean(Xtrain)
        Dtrain -= np.mean(Dtrain)
        
        Xtest  -=np.mean(Xtest)
        Dtest -= np.mean(Dtest)
        
        return Xtrain,Xtest,Dtrain,Dtest
    
    if system == 'reverseSS1':
        trainSeed = np.random.randint(0,500)
        testSeed = np.random.randint(0,500)
        Xtrain,Dtrain = getSS1(systemInfo['N']+1000,systemInfo['L'],np.pi,systemInfo['Powers'],systemInfo['Weights'],seed = trainSeed)
        Xtest,Dtest = getSS1(systemInfo['Ntest']+1000,systemInfo['L'],np.pi,systemInfo['Powers'],systemInfo['Weights'],seed = testSeed)
        
        Xtrain,Dtrain  =getPredInputs(Dtrain,systemInfo['N'],systemInfo['L'],startidx=systemInfo['L']+10,pH=systemInfo['pH'])
        Xtest,Dtest  =getPredInputs(Dtest,systemInfo['Ntest'],systemInfo['L'],startidx=systemInfo['L']+10,pH=systemInfo['pH'])
        
        #Xtrain  -=np.mean(Xtrain)
        #Dtrain -= np.mean(Dtrain)
        
        #Xtest  -=np.mean(Xtest)
        #Dtest -= np.mean(Dtest)
        
        
        return Xtrain,Xtest,Dtrain,Dtest
        
            
    if system =='SSLinear':
        SS = np.load('./Signals/WSS_linear.npy')/10
        trainSet= SS[:10000]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        sTrain = np.random.randint(1500,2000)
        
        
        testSet = SS[10000:20000]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        sTest =np.random.randint(1500,2000)
        
        print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        if systemInfo['align'] == True:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],systemInfo['L'],sTrain,noiseTrain,sdim=systemInfo['sdim'])
            Xtest = Xtrain[systemInfo['N']:]
            dTest = dTrain[systemInfo['N']:]
            Xtrain =Xtrain[:systemInfo['N']]
            dTrain =dTrain[:systemInfo['N']]
        else:
            Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,pH=systemInfo['pH'],sdim=systemInfo['sdim'])
            Xtest,dTest = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,sdim=systemInfo['sdim'])
        
    if system == 'Noise2Poly':
        #Xtrain = np.zeros((systemInfo['N'],systemInfo['L']))
        Xtrain= np.random.uniform(-1,1,size=(systemInfo['N'],systemInfo['L']))
        Xtest = np.random.uniform(-1,1,size=(systemInfo['Ntest'],systemInfo['L']))
        #xmin = np.min(Xtrain[:,0]);xmax=np.max(Xtrain[:,0]);ymin =np.min(Xtrain[:,1]);ymax=np.max(Xtrain[:,1])
        #X, Y = np.mgrid[-1:1:0.05, -1:1:0.05]
        #Xtest = np.vstack([X.ravel(), Y.ravel()]).T
        dTrain = getDesired(Xtrain, systemInfo['orders'], systemInfo['weights']) 
        #dTrain = dTrain - np.mean(dTrain)
        dTest = getDesired(Xtest, systemInfo['orders'], systemInfo['weights'])
        #dTest = dTest - np.mean(dTest)
        
        return Xtrain/10,Xtest/10,dTrain/10,dTest/10
        if systemInfo['Embed']:
            Xtrain,dTrain = getInputEmbed(Xtrain, dTrain)
            Xtest, dTest = getInputEmbed(Xtest, dTest)
            
        
    if system =='Fish':
        df = pd.read_csv(r'./Signals/Fish.csv')
        train,test = train_test_split(df,test_size=0.1,random_state=seed)
        Xtrain = train.drop(['Species','Weight'],axis=1).to_numpy()
        dTrain = train['Weight'].to_numpy()/100
        Xtest = test.drop(['Species','Weight'],axis=1).to_numpy()
        dTest = test['Weight'].to_numpy()/100
        
    
    if system =='SunSpot':
        SN_list_ = []
        with open('./Signals/SN_m_tot_V2.0.txt', 'r') as fd:
          reader = csv.reader(fd)
          for row in reader:
            SN_list = []
            list_ = row[0].split(' ')
            for number in list_:
              try: SN_list.append(float(number))
              except: continue
            SN_list_.append(np.array(SN_list, dtype='float32'))
        data = np.stack(SN_list_)[:, 3]
        mean = np.mean(data)
        std = np.std(data)
        data = (data-mean)/std
        
        
        trainSet = data[0:2750]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainSet))
        if seed == 0:
            sTrain = np.random.randint(30,80)
        else:
            sTrain = np.random.randint(30,80)
        testSet = data[2750:]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testSet))
        if seed == 0:
            sTest =np.random.randint(30,50)
        else:
            sTest =np.random.randint(30,50)
        #print(f'seed:{seed}, sTrain:{sTrain}, sTest:{sTest}')
        Xtrain,dTrain = getPredInputs(trainSet,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,sdim=systemInfo['sdim'],pH=systemInfo['pH'])
        Xtest,dTest = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,sdim=systemInfo['sdim'],pH=systemInfo['pH'])
        
        '''
        trainMean =np.mean(Xtrain[:,0])
        trainStd =np.std(Xtrain[:,0])
        testMean = np.mean(Xtest[:,0])
        testStd = np.std(Xtest[:,0])
        dTrain  = (dTrain - np.mean(Xtrain[:,0]))/np.std(Xtrain[:,0])
        Xtrain = (Xtrain - np.mean(Xtrain[:,0]))/np.std(Xtrain[:,0])
    
        dTest = (dTest-np.mean(Xtest[:,0]))/np.std(Xtest[:,0])
        Xtest = (Xtest-np.mean(Xtest[:,0]))/np.std(Xtest[:,0])
        '''
        
        return Xtrain,Xtest,dTrain,dTest
        
        
    if system == 'sin2square':
        Fs = 10000
        f = 500
        sample = 20000
        x = np.arange(sample)
        data= np.sin(2 * np.pi * f * x / Fs)
        noise =systemInfo['wgnVar']*np.random.randn(len(data))
        data = data+noise
        trainSet =data[:15000]
        testSet = data[15000:]
        d = sig.square((2* np.pi * f* x / Fs),duty=0.5)
        d = np.sin(2*4*np.pi * f * x / Fs)
        dtest = d[1500:]
        Dtrain = d[:15000]
        Dtest = d[15000:]
        Xtrain,_ = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],systemInfo['L'],500,sdim=systemInfo['sdim'])
        Dtrain_lag,_ = getPredInputs(Dtrain,systemInfo['N'],systemInfo['L'],500,sdim=systemInfo['sdim'])
        Xtest,_ = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],300,sdim=systemInfo['sdim'])
        #Xtest = Xtrain[-systemInfo['Ntest']:]
        Xtrain =Xtrain[:-systemInfo['Ntest']]
        dTrain = Dtrain[500:500+systemInfo['N']]
        dTest = Dtest[:systemInfo['Ntest']]
        #dTest = Dtrain[500+systemInfo['N']:500+systemInfo['N']+systemInfo['Ntest']]
        return Xtrain,Xtest,dTrain,dTest#,Dtrain_lag  
        
    if system == 'HenonMap':
        
        Xtrain = getHenonMap(systemInfo['N']+2000)
        Xtest = getHenonMap(systemInfo['Ntest']+2000,startPoint=Xtrain[-1,:])
        noiseTest = systemInfo['wgnVar']*np.random.randn(Xtest.shape[0])
        noiseTrain = systemInfo['wgnVar']*np.random.randn(Xtrain.shape[0])
        sTrain = np.random.randint(100,500)
        sTest= np.random.randint(100,500)
        Xtrain,dTrain  = getPredInputs(Xtrain[:,0], systemInfo['N'], L=systemInfo['L'],noise = noiseTrain,startidx=sTrain,pH=systemInfo['pH'])
        Xtest,dTest  = getPredInputs(Xtest[:,0], systemInfo['Ntest'], L=systemInfo['L'],noise = noiseTest, startidx=sTest,pH=systemInfo['pH'])
        
        
    if system == 'Lorenz3d':
        
        Data = lorenz(10000)[1]
        Data= sig.decimate(Data,2)
        Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
        trainidxs= [np.arange(0,5000)]
        trainidxs = [x for y in trainidxs for x in y ]
        testidxs= np.arange(5000,6000)
        
        
        trainInput = Data[trainidxs,0]
        trainOutput = Data[trainidxs,2]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainInput))
        sTrain = np.random.randint(0,200)
        
        testInput = Data[testidxs,0]
        testOutput = Data[testidxs,2]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testInput))
        sTest = np.random.randint(10,500)
        
        
        Xtrain,dTrain = getInputs(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain,noiseTrain,pH = systemInfo['pH'])
        Xtest,dTest = getInputs(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest,noiseTest,pH = systemInfo['pH'])
        
      
    if system == 'Lorenz3ddB':
        
        Data = lorenz(40000)[1]/10
        Datadec = np.zeros((int(Data.shape[0]/2),Data.shape[1]))
        # Datadec[:, 0] = sig.decimate(Data[:,0], 2)
        # Datadec[:,1] = sig.decimate(Data[:,1], 2)
        # Datadec[:, 2] = sig.decimate(Data[:,2], 2)
        # Data = Datadec
        #Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
       
        trainInput = Data[:15000,0]
        trainOutput = Data[:15000,2]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainInput))
        sTrain = np.random.randint(0,3000)
        
        testInput = Data[15000:,0]
        testOutput = Data[15000:,2]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testInput))
        sTest = np.random.randint(10,2000)
        sTest = 500
        
        
        Xtrain,dTrain = getInputsNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain+systemInfo['L'],systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest,dTest = getInputsNoisedB(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest+systemInfo['L'],systemInfo['wgnVar'],pH = systemInfo['pH'])

    '''
    if system == 'Lorenz3ddB':
        
        DataOG = lorenz(20000)[1]
        Data = np.zeros((10000,3))
        Data[:,0]= sig.decimate(DataOG[:,0],2)
        Data[:,1] = sig.decimate(DataOG[:,1],2)
        Data[:,2]= sig.decimate(DataOG[:,2],2)
        Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
       
        trainInput = Data[:5000,0]
        trainOutput = Data[:5000,2]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainInput))
        sTrain = np.random.randint(systemInfo['L'],2500)
        
        testInput = Data[5500:,0]
        testOutput = Data[5500:,2]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testInput))
        sTest = np.random.randint(systemInfo['L'],3000)
        
        
        Xtrain,dTrain = getInputsNoisedB(trainInput,trainOutput,systemInfo['N'],systemInfo['L'],sTrain,systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        Xtest,dTest = getInputsNoisedB(testInput,testOutput,systemInfo['Ntest'],systemInfo['L'],sTest,systemInfo['wgnVar'],pH = systemInfo['pH'])
     '''
    if system == 'Lorenzxy2z':
        
        Data = lorenz(10000)[1]/10
        Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
        
        
       
        trainInputx = Data[:5000,0]
        trainInputy = Data[:5000,1]
        trainInputz = Data[:5000,2]
        trainOutput = Data[systemInfo['pH']:5000,2]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainInputx))
        sTrain = np.random.randint(32,450)
        
        Xtrainx,dTrain = getInputs(trainInputx, trainOutput, systemInfo['N'], systemInfo['L'], sTrain)
        Xtrainy,dTrain = getInputs(trainInputy, trainOutput, systemInfo['N'], systemInfo['L'], sTrain)
        Xtrainz,dTrain = getInputs(trainInputz, trainOutput, systemInfo['N'], systemInfo['L'], sTrain)
        Xtrain = np.dstack((Xtrainx[:,:,np.newaxis],Xtrainy[:,:,np.newaxis],Xtrainz[:,:,np.newaxis]))
        
        testInputx = Data[5000:,0]
        testInputy = Data[5000:,1]
        testInputz = Data[5000:,2]
        testOutput = Data[5000+systemInfo['pH']:,2]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testInputx))
        sTest = np.random.randint(32,450)
        
        Xtestx,dTest = getInputs(testInputx, testOutput, systemInfo['Ntest'], systemInfo['L'], sTest)
        Xtesty,dTest = getInputs(testInputy, testOutput, systemInfo['Ntest'], systemInfo['L'], sTest)
        Xtestz,dTest = getInputs(testInputz, testOutput, systemInfo['Ntest'], systemInfo['L'], sTest)
        Xtest = np.dstack((Xtestx[:,:,np.newaxis],Xtesty[:,:,np.newaxis],Xtestz[:,:,np.newaxis]))
        
        
        
        
    
    if system == 'LorenzXYZ2Z':
        
        Data = lorenz(10000)[1]
        Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
       
        trainInput = Data[:5000,:]
        trainOutput = Data[:5000,2]
        noiseTrain = systemInfo['wgnVar']*np.random.randn(len(trainInput))
        idxs = np.random.choice(len(trainInput)-1,size = systemInfo['N'])
        Xtrain = trainInput[idxs,:]
        dTrain = trainOutput[idxs+1]
     
        
        testInput = Data[5500:,:]
        testOutput = Data[5500:,2]
        noiseTest = systemInfo['wgnVar']*np.random.randn(len(testInput))
        idxs = np.random.choice(len(testInput)-1,size = systemInfo['Ntest'])
        Xtest= testInput[idxs,:]
        dTest = testOutput[idxs+1]
        
    
    if system == 'LorenzState':
        
        Data = lorenz(10000)[1]
        Data = (Data-np.mean(Data,axis=0))/np.std(Data,axis=0)
        
        
        Xtrain = Data[:5000,:]
        dTrain = Data[1000:6000,:]
        
        Xtest = Data[5500:6000,:]
        dTest = Data[6500:7000,:]
      
        
        
    if system == 'Climate':
        
        scaler = MinMaxScaler((-1,1))
        dfTrain  = pd.read_csv('./Signals/DailyDelhiClimateTrain.csv')
        dfTrain = dfTrain.join(dfTrain['date'].str.split('-',expand=True).rename(columns={0:'year', 1:'month',2:'day'}))
        dfTrain[['year','month','day','humidity','wind_speed','meanpressure']] = scaler.fit_transform(dfTrain[['year','month','day','humidity','wind_speed','meanpressure']])
        dTrain = dfTrain['meantemp'].to_numpy()
        Xtrain = dfTrain[['year','month','day','humidity','wind_speed','meanpressure']].to_numpy()
        
        dfTest  = pd.read_csv('./Signals/DailyDelhiClimateTest.csv')
        dfTest = dfTest.join(dfTest['date'].str.split('-',expand=True).rename(columns={0:'year', 1:'month',2:'day'}))
        dfTest[['year','month','day','humidity','wind_speed','meanpressure']] = scaler.fit_transform(dfTest[['year','month','day','humidity','wind_speed','meanpressure']])
        dTest =dfTest['meantemp'].to_numpy()
        Xtest = dfTest[['year','month','day','humidity','wind_speed','meanpressure']].to_numpy()
        
    
    if system =='BikeSharing':
        bike_sharing = fetch_ucirepo(id=275) 
        names  = np.array(bike_sharing.variables['name'])
        names = np.delete(names,[0,11,5,14,15,16],axis=0)
        # data convert to numpy and min-max normalize 
        XX = np.array(bike_sharing.data.features)
        intDates = [int(''.join(x.split('-'))) for x in XX[:,0]]
        XX[:,0] = intDates
        XX = np.delete(XX,0,axis=1)
        XX = np.delete(XX,10,axis=1)
        XX = XX.astype(float)
        YY = np.array(bike_sharing.data.targets)
        YY = YY.astype(float)


        XX,miniX,maxiX = minMaxNorm(XX)
        YY,miniY,maxiY = minMaxNorm(YY)
        
        x,xtest,y,ytest = train_test_split(XX,YY,test_size=systemInfo['testRatio'], random_state=seed)
        y= np.squeeze(y);ytest= np.squeeze(ytest)
        
        return x,xtest,y,ytest,miniX,maxiX,miniY,maxiY,names
        
        
    
    if system == 'RealEstate':
    
        real_estate_valuation = fetch_ucirepo(id=477) 
        names =real_estate_valuation.data.features.columns
        names = [" ".join(names[l].split(" ")[1:]) for l in range(len(names))]
        #print(f'Covariates: {names}')
          
        # data convert to numpy and min-max normalize 
        XX = np.array(real_estate_valuation.data.features)
        #print(f'Pre Normilzation Data example: {XX[0,:]}')
        YY = np.array(real_estate_valuation.data.targets)
        XX,miniX,maxiX = minMaxNorm(XX)
        YY_temp,miniY,maxiY = minMaxNorm(YY)
        
        x,xtest,y,ytest = train_test_split(XX,YY,test_size=systemInfo['testRatio'], random_state=seed)
        y= np.squeeze(y);ytest= np.squeeze(ytest)
        outlier = np.argmax(y)
        x = np.delete(x,outlier,axis=0)
        y = np.delete(y,outlier,axis=0)
        
        return x,xtest,y,ytest,miniX,maxiX,miniY,maxiY,names
        
    if system == 'Concrete':
        concrete_compressive_strength = fetch_ucirepo(id=165) 
  
        # data (as pandas dataframes) 
        X = np.array(concrete_compressive_strength.data.features)
        y = np.array(concrete_compressive_strength.data.targets)
        
        XX,miniX,maxiX = minMaxNorm(X)
        
        x,xtest,y,ytest = train_test_split(XX,y,test_size=systemInfo['testRatio'], random_state=seed)
        
        y = np.squeeze(y);ytest = np.squeeze(ytest)
        
        return x,xtest,y,ytest
    
    if system == 'Births':
        
        dataset  = load_dataset("monash_tsf", "us_births")
        trainSignal = np.squeeze(np.array(dataset['train']['target']).T)
        testSignal = np.squeeze(np.array(dataset['test']['target']).T)
        trainSignal,miniX,maxiX = minMaxNorm(trainSignal)
        testSignal= aminMaxNorm(testSignal,miniX,maxiX)
        trainsidx = np.random.randint(systemInfo['L']+1,1000)
        testidx = np.random.randint(systemInfo['L']+1,1000)
        x,d = getPredInputs(trainSignal,systemInfo['N'],systemInfo['L'],trainsidx,noise = 0,pH=systemInfo['pH'],sdim=False)
        xtest,dtest = getPredInputs(testSignal,systemInfo['Ntest'],systemInfo['L'],testidx,noise = 0,pH=systemInfo['pH'],sdim=False)
        
        return x,xtest,d,dtest
        

    if system == 'Mines':
        # fetch dataset 
        connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
          
        # data (as pandas dataframes) 
        X = np.array(connectionist_bench_sonar_mines_vs_rocks.data.features)
        y = connectionist_bench_sonar_mines_vs_rocks.data.targets 
        y=y.replace('R', 0.)
        y=y.replace('M',1.)
        y = np.array(y)
       
        
        
        XX,miniX,maxiX = minMaxNorm(X)
        x,xtest,y,ytest = train_test_split(XX,y,test_size=systemInfo['testRatio'], random_state=seed)
        y = np.squeeze(y);ytest = np.squeeze(ytest)
        return x,xtest,y,ytest

    if system == 'PJMAEP':
        df = pd.read_csv('/media/benjamin/5CCEC71CCEC6ECF8/KUBUTU/NoTrick/Signals/PJMData/AEP_hourly.csv')
        dataset = np.array(df['AEP_MW'])
        #dataset,miniX,maxiX = minMaxNorm(dataset)
        dataset = (dataset-np.mean(dataset))/np.std(dataset)
        trainSignal = dataset[40000:80000]
        valSignal = dataset[80000:100000]
        testSignal = dataset[100000:]
        trainsidx = np.random.randint(systemInfo['L']+1,5000)
        testidx = np.random.randint(systemInfo['L']+1,15000)
        testidx = 5000
        validx = 5000
        x,d=getInputsJumpNoisedB(trainSignal,trainSignal,systemInfo['N'],systemInfo['L'],
                                trainsidx,E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        x_val,d_val=getInputsJumpNoisedB(valSignal,valSignal,systemInfo['Ntest'],systemInfo['L'],validx,
                                   E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
        xtest,dtest=getInputsJumpNoisedB(testSignal,testSignal,systemInfo['Ntest'],systemInfo['L'],testidx,
                                         E=systemInfo['D'],noisedb=systemInfo['wgnVar'],pH = systemInfo['pH'])
                      
        #x,d = getPredInputs(trainSignal,systemInfo['N'],systemInfo['L'],trainsidx,noise = 0,pH=systemInfo['pH'],sdim=False)
        #xtest,dtest = getPredInputs(testSignal,systemInfo['Ntest'],systemInfo['L'],testidx,noise = 0,pH=systemInfo['pH'],sdim=False)
        
        
        return x,x_val,xtest,d,d_val,dtest
    
        
    
    if system == 'Telescope':
        # fetch dataset 
        magic_gamma_telescope = fetch_ucirepo(id=159) 
  
        # data (as pandas dataframes) 
        X = np.array(magic_gamma_telescope.data.features)
        y = magic_gamma_telescope.data.targets
        y=y.replace('g',1)
        y=y.replace('h',0)
        y = np.array(y)
        
        
          
        XX,miniX,maxiX = minMaxNorm(X)
        x,xtest,y,ytest = train_test_split(XX,y,test_size=systemInfo['testRatio'], random_state=seed)
        y = np.squeeze(y);ytest = np.squeeze(ytest)
        return x,xtest,y,ytest
    
    if system == 'Yeast':
        # fetch dataset 
        yeast = fetch_ucirepo(id=110) 
          
        # data (as pandas dataframes) 
        X = np.array(yeast.data.features)[:,1:]
        
        y = yeast.data.targets
        classes = y['localization_site'].unique()
        for i,c in enumerate(classes):
            y=y.replace(f'{c}',float(i))
        y = np.array(y)
        
        
          
        XX,miniX,maxiX = minMaxNorm(X)
        x,xtest,y,ytest = train_test_split(XX,y,test_size=systemInfo['testRatio'], random_state=seed)
        y = np.squeeze(y);ytest = np.squeeze(ytest)
        return x,xtest,y,ytest
    
    if system == 'NSID':
        '''
        Kernel size is 0.75
        '''
        xx = np.load('./Signals/x_nsid.npy')[:,0]
        xxtest = np.load('./Signals/xtest_nsid.npy')[:,0]
        dd = np.load('./Signals/d_nsid.npy')
        ddtest = np.load('./Signals/dtest_nsid.npy')
        Xtrain, dTrain = getInputsNoisedB(xx,dd,systemInfo['N'],systemInfo['L'],1+systemInfo['L'],systemInfo['wgnVar'],pH = systemInfo['pH'])
        Xtest, dTest = getInputsNoisedB(xxtest,ddtest,systemInfo['Ntest'],systemInfo['L'],1+systemInfo['L'],systemInfo['wgnVar'],pH = systemInfo['pH'])
        
        
        if systemInfo['AlphaStable']:
            dnoise = levy_stable.rvs(alpha=systemInfo['Stable_Alpha'], beta=0.,loc=0.,scale=systemInfo['Stable_Scale'], size=len(dTrain))
            dclean = np.copy(dTrain)
            dTrain +=dnoise
            return Xtrain,Xtest,dTrain,dTest,dclean
        
    if system == 'NSID2':
        np.random.seed(seed)
        if systemInfo['AlphaStable']:
            xgen = np.zeros(10000)
            xgen[0]  = 0.5
            xgen[1] = 0.1
            noise = levy_stable.rvs(alpha=systemInfo['Stable_Alpha'], beta=0.,loc=0.,scale=systemInfo['Stable_Scale'], size=10000) 
            for n in range(2,10000):
                xgen[n] = ((0.8-0.5*np.exp(-xgen[n-1]**2))*(xgen[n-1]))-((0.3 +0.9*np.exp(-xgen[n-1]**2))*(xgen[n-2])) + (0.1*np.sin(3.1415926*xgen[n-1])) +noise[n]
            xgen =xgen[100:]
            
            xtestgen = np.zeros(5000)
            xtestgen[0]  =0.5
            xtestgen[1] = 0.1
            for n in range(2,5000):
                xtestgen[n] = ((0.8-0.5*np.exp(-xtestgen[n-1]**2))*(xtestgen[n-1]))-((0.3 +0.9*np.exp(-xtestgen[n-1]**2))*(xtestgen[n-2])) + (0.1*np.sin(3.1415926**xtestgen[n-1]))
            xtestgen =xtestgen[100:]
            
            Xtrain,dTrain = getPredInputsNoisedB(xgen,N=systemInfo['N'],L=systemInfo['L'],startidx=systemInfo['L']+1,noisedb = systemInfo['wgnVar'],pH=systemInfo['pH'],sdim=False)
            Xtest,dTest = getPredInputsNoisedB(xtestgen,N=systemInfo['Ntest'],L=systemInfo['L'],startidx=systemInfo['L']+1+200,noisedb = systemInfo['wgnVar'],pH=systemInfo['pH'],sdim=False)
            return Xtrain,Xtest,dTrain,dTest,dTrain
        else:
            xgen = np.zeros(10000)
            xgen[0]  = 0.
            xgen[1] = 0.
            noise = np.random.normal(0,systemInfo['wgnVar'],10000)
            for n in range(2,10000):
                xgen[n] = ((0.8-0.5*np.exp(-xgen[n-1]**2))*(xgen[n-1]))-((0.3 +0.9*np.exp(-xgen[n-1]**2))*(xgen[n-2])) + (0.1*np.sin(np.pi*xgen[n-1])) +noise[n]
            xgen =xgen[100:]
            
            xtestgen = np.zeros(5000)
            xtestgen[0]  =0.5
            xtestgen[1] = 0.1
            for n in range(2,5000):
                xtestgen[n] = ((0.8-0.5*np.exp(-xtestgen[n-1]**2))*(xtestgen[n-1]))-((0.3 +0.9*np.exp(-xtestgen[n-1]**2))*(xtestgen[n-2])) + (0.1*np.sin(np.pi*xtestgen[n-1]))
            xtestgen =xtestgen[100:]
            
            trainstart = np.random.randint(0,5000)
            teststart = np.random.randint(0,2500)
            Xtrain,dTrain = getPredInputsNoisedB(xgen,N=systemInfo['N'],L=systemInfo['L'],startidx=systemInfo['L']+trainstart,noisedb = systemInfo['wgnVar'],pH=systemInfo['pH'],sdim=False)
            Xtest,dTest = getPredInputsNoisedB(xtestgen,N=systemInfo['Ntest'],L=systemInfo['L'],startidx=systemInfo['L']+teststart,noisedb = systemInfo['wgnVar'],pH=systemInfo['pH'],sdim=False)
            return Xtrain,Xtest,dTrain,dTest,dTrain
            
        
        
        
        
    
    return Xtrain,Xtest,dTrain,dTest 

def WSNR(w0,wtrue):
    num = wtrue.T@wtrue
    den = (wtrue-w0).T@(wtrue-w0)
    return 10*np.log10(num/den)

def createInputsEmbed(systemInfo,E,seed=0):
    system = systemInfo['System']
    
    if system == 'sin2square':
        Fs = 10000
        f = 1000
        sample = 20000
        x = np.arange(sample)
        data= np.sin(2 * np.pi * f * x / Fs)
        noise =systemInfo['wgnVar']*np.random.randn(len(data))
        data = data+noise
        trainSet =data[:15000]
        testSet = data[15000:]
        #d = sig.square(2*2* np.pi * f* x / Fs,duty=0.5)
        d = np.sin(2*2*np.pi * f * x / Fs)
        Dtrain = d[:15000]
        Dtest = d[15000:]
        Xtrain,_ = getPredInputs(trainSet,systemInfo['N']+systemInfo['Ntest'],systemInfo['L'],500,sdim=systemInfo['sdim'])
        Dtrain_lag,_ = getPredInputs(Dtrain,systemInfo['N'],systemInfo['L'],500,sdim=systemInfo['sdim'])
        #Xtest,_ = getPredInputs(testSet,systemInfo['Ntest'],systemInfo['L'],300,sdim=systemInfo['sdim'])
        Xtest = Xtrain[-systemInfo['Ntest']:]
        Xtrain =Xtrain[:-systemInfo['Ntest']]
        Xtrain = getInputEmbed(Xtrain,E)
        dTrain = Dtrain[500:500+systemInfo['N']]
        dTest = Dtrain[500+systemInfo['N']:500+systemInfo['N']+systemInfo['Ntest']]
        return Xtrain,Xtest,dTrain,dTest
    
    
def getInputEmbed(X,D,E=None):
    N = X.shape[0]
    L = X.shape[1]
    if E == None:
        E = L
    X_em = np.zeros((N-L,L,E))
    d_em = np.zeros(N-L)
    for n in range(L,N):
        X_em[n-L,:,:] = np.flip(X[n-L:n,:E],axis=0)
        d_em[n-L] = D[n]
    
    return X_em,d_em
    
    
def getDesired(X,orders,weights):
    '''
    generates desired signal specified by input, orders, and weights.

    Parameters
    ----------
    X : np array
        Input vectors must.(NxL) L should be atleast 2
    order : np array,
        powers applied to each dimension of the input vectors.(Lx1)
    weights : np array, 
        weight applied to each dimension of the input vectors.(Lx1)(#s between 0-1) 

    Returns
    -------
    D : np array
        Desired response at input locations.(Nx1)

    '''
    D = (X**orders)@weights
    
    return D



def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def generate_SUN(order = 10):
  SN_list_ = []
  with open('./drive/MyDrive/SN_m_tot_V2.0.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
      SN_list = []
      list_ = row[0].split(' ')
      for number in list_:
        try: SN_list.append(float(number))
        except: continue
      SN_list_.append(np.array(SN_list, dtype='float32'))
  data = np.stack(SN_list_)[:, 3]
  x, desire = construct_time_windows(data[:-1], data[1:], order)
  return x, desire

def getPredInputs(signal,N,L,startidx,noise = 0,pH=1,sdim=False):
    
    t = 0
    x = np.zeros((N,L))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    signalN = signal+noise
    
    for i in range(N):
        x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
        if sdim:
            d[i] = signal[t+pH+startidx]
        else:
            d[i] = signal[t+pH+startidx]
        t +=1
    if sdim:
        N = len(d)
        L = x.shape[1]
        x1d = np.zeros(N+L)
        for i in range(0,N,L):
            if i == N-L:
                stop=1
            x1d[i:i+L] = np.flip(x[i,:])
        
        x1d[-L:] = np.flip(x[-1,:])
        return x1d,d
      
    
    return x,d

def getPredInputsNoisedB(signal,N,L,startidx,noisedb = 0,pH=1,sdim=False):
    
    t = 0
    x = np.zeros((N,L))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    if noisedb != 0 or noisedb != 100:
        nsig = np.sqrt(np.mean(signal**2)*(10**(-noisedb/20)))
        noise  = np.random.normal(0,nsig,len(signal))
        #print(f'noiseStd = {nsig}')
        signalN = signal + noise
        signalN  -= np.mean(signalN)
    else:
        signalN = signal -np.mean(signal)
    
    for i in range(N):
        x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
        if sdim:
            d[i] = signal[t+pH+startidx]
        else:
            d[i] = signalN[t+pH+startidx]
        t +=1
    if sdim:
        N = len(d)
        L = x.shape[1]
        x1d = np.zeros(N+L)
        for i in range(0,N,L):
            if i == N-L:
                stop=1
            x1d[i:i+L] = np.flip(x[i,:])
        
        x1d[-L:] = np.flip(x[-1,:])
        return x1d,d
    
    return x,d


def getInputs(Input,Desired,N,L,startidx,noise = 0,pH=0,sdim=False):
    
    t = 0
    x = np.zeros((N,L))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    signalN =  Input+noise
    
    for i in range(N):
        x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
        d[i] = Desired[t+pH+startidx]
        t +=1
      
    return x,d

def getInputsNoisedB(Input,Desired,N,L,startidx,noisedb = 0,pH=0,sdim=False,zeropad=False):
    
    t = 0
    x = np.zeros((N,L))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    if noisedb != 0 or noisedb != 100:
        nsig = np.sqrt(np.mean(Input**2)*(10**(-noisedb/20)))
        noise  = np.random.normal(0,nsig,len(Input))
        #print(f'noiseStd = {nsig}')
        signalN = Input + noise
    else:
        signalN = Input
    if zeropad:
        signalN = np.pad(signalN, (L, 0), 'constant', constant_values=(0, 0))
        startidx=L
        for i in range(N):
            x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH]
            t +=1
        
    else:
        for i in range(N):
            x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH+startidx]
            t +=1
    
      
    return x,d

def getInputsnoNoise(Input,Desired,N,L,startidx,noisedb = 0,pH=0,sdim=False,zeropad=False):
    
    t = 0
    x = np.zeros((N,L))
    d = np.zeros(N)
    #signal = (signal-np.mean(signal))
    
    signalN = Input
    if zeropad:
        signalN = np.pad(signalN, (L, 0), 'constant', constant_values=(0, 0))
        startidx=L
        for i in range(N):
            x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH]
            t +=1
        
    else:
        for i in range(N):
            x[i,:] = np.flip(signalN[t-L+1+startidx:t+1+startidx])
            d[i] = Desired[t+pH+startidx]
            t +=1
    
      
    return x,d





def matrix_equilibrate(A, p=2):
    '''
    Matrix equilibration. taken from https://github.com/isledge/MBRCE/blob/main/MB-RenyisCrossEntropy.ipynb

    Parameters
    ----------
    A : np array
        Matrix to be equilibrated.
    p : int, optional
        Power used for equilibration. The default is 2.

    Returns
    -------
    np array
        Equilibrated version of A.

    '''
    B = np.power(np.abs(A), p)

    obj = 0
    u = cvxpy.Variable(A.shape[0])
    v = cvxpy.Variable(A.shape[1])

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            obj += cvxpy.exp(cvxpy.log(B[i,j]) + u[i] + v[j])

    obj = cvxpy.Minimize(obj)
    constraints = [sum(u)==0, sum(v)==0]
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(verbose=False)
    
    D = np.diagflat(np.exp(u.value/p))
    E = np.diagflat(np.exp(v.value/p))
    
    return D * A * E

def TrainPredGif(IP,x,d,p,K=10,name='.'):
    '''
    

    Parameters
    ----------
    IP : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    saveDir = f'./Plots/GIFs/{name}'
    if os.path.isdir(saveDir)==False:
        os.mkdir(saveDir)
    N = len(IP)
    for n in range(N):
        ip = IP[n]
        idxs = np.argsort(ip)[::-1]
        idxs=idxs[:K] 
        ip_k = ip[idxs]
        plt.figure(dpi=300)
        plt.plot(d,label='Training Signal')
        plt.scatter(idxs,d[idxs],c=ip_k,edgecolors='r',label='weighted samples')
        plt.scatter(n,d[n],label='desired value',marker='x',color='m')
        plt.scatter(n,p[n],label='prediction',marker='x',facecolors='c')
       
        
        plt.colorbar()
        plt.savefig(f'{saveDir}/{n}.png')
        plt.close()



def TestPredGif(IP,xtest,dTest,xTrain,p,K=10,name='.'):
    '''
    

    Parameters
    ----------
    IP : TYPE
        DESCRIPTION.
    xtest : TYPE
        DESCRIPTION.
    dTest : TYPE
        DESCRIPTION.
    dTrain : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    K : TYPE, optional
        DESCRIPTION. The default is 10.
    name : TYPE, optional
        DESCRIPTION. The default is '.'.

    Returns
    -------
    None.

    '''
    saveDir = f'./Plots/GIFs/{name}'
    if os.path.isdir(saveDir)==False:
        os.mkdir(saveDir)
    N = len(IP)
    for n in range(N):
        ip = IP[n]
        idxs = np.argsort(ip)[::-1]
        idxs=idxs[:K] 
        ip_k = normalizeData(ip[idxs])
        fig, axs =plt.subplots(2,dpi=300)
        axs[0].plot(xTrain[:,0],label='Training Signal')
        ws = axs[0].scatter(idxs,xTrain[idxs,0],c=ip_k,s=10,label='weighted samples',alpha=1)
        axs[0].legend()
        axs[1].plot(xtest[:,0],label='Testing Signal')
        axs[1].scatter(n,xtest[n,0],marker='x',color='c',label='Input Sample')
        axs[1].scatter(n+1,dTest[n],marker='x',color='m',label='Desired Value')
        axs[1].scatter(n+1,p[n],marker='x',color='r',label='Prediction')
        axs[1].legend()
        plt.colorbar(ws, ax=axs[0])
        plt.savefig(f'{saveDir}/{n}.png')
        plt.close()
        
def input2IP(X,Y,sigma,kt):
    N = len(X)
    L = Y.shape[1]
    X_IP = np.zeros((N,L))
    for n in range(N):
        X_IP[n,:] = np.mean(itl.GK(X[n,:],Y,sigma,kernelType=kt),axis=0)
    
    return X_IP

def getTrainMap(x,d,L,N,K):
    tps = np.zeros((L,L))
    for l in range(L):
        v = np.ones(L)* (1/L)
        tps[l,:] = v *np.max(np.linalg.norm(x,2,axis=1))


    dtp = np.zeros((len(x),L))
    for l in range(L):
        dtp[:,l] = np.linalg.norm(x-tps[l,:],ord=2,axis=1)
    dtp = np.max(dtp,axis=1)
    trainMap ={}
    trainMapTest ={}

    idxs = np.argsort(dtp)
    x_sort = x[idxs,:]
    dtp_sort = dtp[idxs]
    d_sort = d[idxs]

    for i in range(N):
        xt = x_sort[i,:]
        wge = np.linalg.norm(xt-x_sort,ord=2,axis= 1)
        dis  = np.max(np.linalg.norm(xt-tps,ord=2,axis=1))
        disComp = abs(dis-dtp_sort)
        Kidx = np.argsort(disComp)
        KidxTest = np.argsort(wge)
        trainMap[f'{i}'] = [x_sort[KidxTest[:K],:],d_sort[KidxTest[:K]]]
        trainMapTest [f'{i}'] = [x_sort[KidxTest[:K],:],d_sort[KidxTest[:K]]]
        
    return trainMap,dtp_sort,tps,trainMapTest,x_sort



def henon_attractor(x, y, a=1.4, b=0.3):
	'''Computes the next step in the Henon 
	map for arguments x, y with kwargs a and
	b as constants.
	'''
	x_next = 1 - a * x ** 2 + y
	y_next = b * x
	return x_next, y_next

def getHenonMap(N,a=1.1, b=0.3,startPoint = [0,0]):
        # number of iterations and array initialization

    X = np.zeros((N,2))
    X[0,:] = startPoint
    
    # add points to array
    for i in range(N-1):
    	x_next, y_next = henon_attractor(X[i,0], X[i,1])
    	X[i+1,0] = x_next
    	X[i+1,1]= y_next
        
    return X

def genSS(N,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    Order = len(Powers)
    sos = butter(5, 15, 'hp', fs=1000, output='sos')
    x = np.random.normal(0,std,N)
    x = sosfilt(sos, x)
    imp = np.array([1,0.5,0.3,0.4])
    imp /=np.sum(imp)
    x =sig.convolve(x,imp,mode='same')
    D = np.zeros(N)
    D[:Order] = x[:Order]
    for n in range(Order,N):
        for l in range(Order):
            D[n] += Weights[l]*x[n-l]**Powers[l]
    
    x = x[Order:]
    D = D[Order:]
    
    return x,D


def getSS(N,L,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    np.random.seed(seed)
    N2 = 10000
    x,D = genSS(N2,std,Powers,Weights)
    sTrain = np.random.randint(2*L,100)
    Xtrain,Dtrain = getInputs(x, D, N, L, sTrain)
    
    
    return Xtrain,Dtrain

def getSS1(N,L,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    np.random.seed(seed)
    N2 = 30000
    x,D = genSS1(N2,std,Powers,Weights)
    sTrain = np.random.randint(2*L,100)
    Xtrain,Dtrain = getInputs(x, D, N, L, sTrain)
    
    
    return Xtrain,Dtrain

def getSSE2(N,L,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    np.random.seed(seed)
    N2 = 30000
    x,D = genSSE2(N2,std,Powers,Weights)
    sTrain = np.random.randint(2*L,100)
    Xtrain,Dtrain = getInputs(x, D, N, L, sTrain)
    
    
    return Xtrain,Dtrain


def genSS1(N,std,Powers=[2,3,3,2,2],Weights=[0.5,1,0.5,0.2,0.75],seed =0):
    Order = len(Powers)
    sos = butter(2,600, 'lowpass', fs=2000, output='sos')
    x = np.random.normal(0,std,N)
    #x = sosfilt(sos, x)
    #imp = np.array([1,0.5,0.3,0.4])
    #imp /=np.sum(imp)
    #xx =sig.convolve(x,imp,mode='same')
    #xx=x
    xx = sosfilt(sos,x)
    D = np.zeros(N)
    D[:Order] = x[:Order]
    for n in range(Order,N):
        for l in range(Order):
            if l %2==0:
                D[n] += Weights[l]*np.tanh(xx[n-l])**Powers[l]
            else:
                D[n] += Weights[l]*np.sin(xx[n-l])**Powers[l]
    
    x = xx[Order:]
    D = D[Order:]
    
    return x,D

def genSSE2(N,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    Order = len(Powers)
    #sos = butter(5, 15, 'hp', fs=1000, output='sos')
    x = np.random.normal(0,std,N)
    D = np.zeros(N)
    D[:Order] = x[:Order]
    for n in range(Order,N-2):
        for l in range(Order):
            if l %2==0:
                D[n] += Weights[l]*(np.tanh(x[n-l])+np.tanh(x[n-(l-1)]))
            else:
                D[n] += Weights[l]*(np.sin(x[n-l]) + np.sin(x[n-(l-1)]))
    
    x = x[Order:]
    D = D[Order:]
    
    return x,D

def genSS2(N,std,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    Order = len(Powers)
    #sos = butter(5, 15, 'hp', fs=1000, output='sos')
    x = np.random.normal(0,std,N)
    #x = sosfilt(sos, x)
    imp = np.array([1,0.5,0.3,0.4])
    imp /=np.sum(imp)
    xx =sig.convolve(x,imp,mode='same')
    D = np.zeros(N)
    D[:Order] = x[:Order]
    for n in range(Order,N):
        for l in range(Order):
            if l %2==0:
                D[n] += Weights[l]*np.sin(xx[n-l])
            else:
                D[n] += Weights[l]*np.sin(xx[n-l])
    
    x = x[Order:]
    D = D[Order:]
    
    return x,D
    
    
def genSS1Test(N,std,limp,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    Order = len(Powers)
    #sos = butter(5, 15, 'hp', fs=1000, output='sos')
    gn = np.random.normal(0,std,N)
    #x = sosfilt(sos, x)
    imp = np.array([1,0.5,0.3,0.4])
    imp /=np.sum(imp)
    xx =sig.convolve(gn,imp,mode='same')
    xxl =sig.convolve(gn,limp,mode='same')
    D = np.zeros(N)
    D[:Order] = xx[:Order]
    for n in range(Order,N):
        for l in range(Order):
            if l %2==0:
                D[n] += Weights[l]*np.tanh(xx[n-l])**Powers[l]
            else:
                D[n] += Weights[l]*np.sin(xx[n-l])**Powers[l]
    
    xx = xx[Order:]
    xxl = xxl[Order:]
    D = D[Order:]
    
    return xx,D


def getSS1Test(N,L,std,limp,Powers=[2,2,3,2,2],Weights=[0.5,1,1,0.2,0.75],seed =0):
    np.random.seed(seed)
    N2 = 30000
    x,xxl,D = genSS1Test(N2,std,limp,Powers,Weights)
    sTrain = np.random.randint(2*L,100)
    Xtrain,Dtrain = getInputs(x, D, N, L, sTrain)
    #FWFtrain,_ = getInputs(xxl, D, N, L, sTrain)
    
    
    return Xtrain,Dtrain


def getImpnoise(N,p0=0.9,impmu=0.2,gstd=0.001):
    choice = np.random.choice([0,1],N,p=[p0,1-p0]) #choice of which gaussian to sample from with probabilities 0.9 0.1
    noise = np.zeros(N)
    for n in range(N):
        #Loop through samples and sample from the Gaussian designated in the "choice" variable.
        if choice[n] == 0:
            noise[n]  = np.random.normal(0,np.sqrt(gstd))
        else:
            noise[n]  = np.random.normal(impmu,np.sqrt(0.01))
            
    return noise
        

        
        
        

