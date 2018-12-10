# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:02:46 2018

@author: lkusters
"""

def verifyhistograms(Ncells,Kobs,Lobs,l1,l2,theta,T0,T1,NX):
    # CHECK THAT THE HISTOGRAM SEEMS OK
    import srampufparest
    from scipy.stats import norm
    import numpy as np 
    
    print('\nTEST 1: VERIFY HISTOGRAMS\n')

    dT = abs(T0-T1)
    M,D = srampufparest.generateSRAMPUFparameters(Ncells,l1,l2,theta)
    merged = srampufparest.generateSRAMPUFobservations(M,D,Kobs,0)
    merged2 = srampufparest.generateSRAMPUFobservations(M,D,Lobs,dT)
    M = np.reshape(M,[Ncells])
    D = np.reshape(D,[Ncells])
    binsc = [i for i in range(Kobs+1)]
    bin_edges = [i-0.5 for i in range(Kobs+2) ]
    hist1d, _ = np.histogram(np.round((1-norm.cdf(0,M,1))*Kobs),bins=bin_edges)
    pKones, Kones = srampufparest.generateSRAMPUFkonesdistribution(Kobs,l1,l2,NX)
    
    phist1d = hist1d/(Ncells)
    print('Verify generated histogram one-probs vs expected histogram at T0, MSE = {0}'.format(np.mean((phist1d-np.asarray(pKones))**2)))
    hist1d,binsc = srampufparest.getcounts1D(merged)
    phist1d = hist1d/(Ncells)
    print('Verify generated histogram cell-observations vs expected histogram at T0, MSE = {0}'.format(np.mean((phist1d-np.asarray(pKones))**2)))
    
    binsc1 = [i for i in range(Kobs+1)]
    binsc2 = [i for i in range(Lobs+1)]
    x_edges = [i-0.5 for i in range(Kobs+2)]
    y_edges = [i-0.5 for i in range(Lobs+2)]
    hist2d, _, _ = np.histogram2d(np.round((1-norm.cdf(0,M,1))*Kobs),np.round((1-norm.cdf(0,M+D*dT,1))*Lobs),bins=[x_edges, y_edges])
    pKLones, Kones, Lones = srampufparest.generateSRAMPUFklonesdistribution(Kobs,Lobs,dT,l1,l2,theta,NX)
    
    phist2d = hist2d/(Ncells)
    print('Verify generated histogram one-probs vs expected histogram T0,T1, difference = {0}'.format(np.mean(np.mean((phist2d-np.asarray(pKLones))**2))))
    hist2d,binsc1,binsc2 = srampufparest.getcounts2D(merged,merged2)
    phist2d = hist2d/(Ncells)
    print('Verify generated histogram cell-observations vs expected histogram T0,T1, difference = {0}'.format(np.mean(np.mean((phist2d-np.asarray(pKLones))**2))))

def verifyestimator_noT(Ncells,Kobs,l1,l2,T0,NX):
    # CHECK THAT THE ESTIMATOR SEEMS OK
    import srampufparest
    import time
    import numpy as np 
    
    print('\nTEST 2: VERIFY ESTIMATOR WITHOUT TEMPERATURE l1={0},l2={1}\n'.format(l1,l2))
    
    Lambdas1 = [np.round(i+l1,2) for i in np.linspace(-0.05,0.05,9)] 
    Lambdas2 = [np.round(i+l2,4) for i in np.linspace(-0.03,0.03,5)] 
    #Thetas = [np.round(i+theta,1) for i in np.linspace(-30,30,7)]
    
    pKones, Kones = srampufparest.generateSRAMPUFkonesdistribution(Kobs,l1,l2,NX)
    t = time.time()
    LL = srampufparest.loop_loglikelihood(np.round(np.asarray(pKones)*Ncells),Kones,NX, Lambdas1, Lambdas2)
    print('time elapsed: {0}'.format(time.time()-t))
    filename = 'testLL_noT'
    srampufparest.writeloglikelihoods(filename+'.txt',LL,NX,Lambdas1,Lambdas2,None)
    LL,NX,Lambdas1,Lambdas2,_ = srampufparest.readloglikelihoods(filename+'.txt')
    j,k = np.unravel_index(LL.argmax(), LL.shape)
    print('l1 = {0}, l2 = {1}'.format(Lambdas1[j],Lambdas2[k]))
    
def verifyestimator_withTsingleloop(Ncells,Kobs,l1,l2,theta,T0,T1,NX):
    # CHECK THAT THE ESTIMATOR SEEMS OK
    import srampufparest
    import time
    import numpy as np 
    
    print('\nTEST 3: VERIFY ESTIMATOR FOR THETA ONLY l1={0},l2={1},theta={2}\n'.format(l1,l2,theta))
    
    dT = abs(T0-T1)
    Thetas = [np.round(i+theta,1) for i in np.linspace(-30,30,19)]
    
    pKLones, Kones, Lones = srampufparest.generateSRAMPUFklonesdistribution(Kobs,Lobs,dT,l1,l2,theta,NX)
    t = time.time()
    LL = srampufparest.loop_loglikelihood_temperature_givenl1l2(np.round(np.asarray(pKLones)*Ncells),Kones,Lones,dT,NX, l1, l2, Thetas)
    LL = np.asarray(LL)
    print('time elapsed: {0}'.format(time.time()-t))
    i = LL.argmax()
    print('theta = {0}, LL = {1}'.format(Thetas[i],LL[i]))
    print(Thetas)
    print(LL)
    
def verifyestimator_withT_doubleloop(Ncells,Kobs,l1,l2,theta,T0,T1,NX):
    # CHECK THAT THE ESTIMATOR SEEMS OK
    import srampufparest
    import time
    import numpy as np 
    
    print('TEST 4: VERIFY ESTIMATOR WITH TEMPERATURE l1={0},l2={1}'.format(l1,l2))
    
    dT = abs(T0-T1)
    Lambdas1 = [np.round(i+l1,2) for i in np.linspace(-0.05,0.05,9)] 
    Lambdas2 = [np.round(i+l2,4) for i in np.linspace(-0.03,0.03,5)] 
    Thetas = [np.round(i+theta,1) for i in np.linspace(-30,30,7)]
    
    pKLones, Kones, Lones = srampufparest.generateSRAMPUFklonesdistribution(Kobs,Lobs,dT,l1,l2,theta,NX)
    t = time.time()
    LL = srampufparest.loop_loglikelihood_temperature_givenl1l2(np.round(np.asarray(pKLones)*Ncells),Kones,Lones,dT,NX, l1, l2, Thetas)
    LL = srampufparest.loop_loglikelihood_temperature(np.round(np.asarray(pKLones)*Ncells),Kones,Lones,dT,NX, Lambdas1, Lambdas2, Thetas)
    print('time elapsed: {0}'.format(time.time()-t))
    filename = 'testLL_T'
    srampufparest.writeloglikelihoods(filename+'.txt',LL,NX,Lambdas1,Lambdas2,Thetas)
    LL,NX,Lambdas1,Lambdas2,Thetas = srampufparest.readloglikelihoods(filename+'.txt')
    i,j,k = np.unravel_index(LL.argmax(), LL.shape)
    print('l1 = {0}, l2 = {1}, theta = {2}'.format(Lambdas1[j],Lambdas2[k],Thetas[i]))


# settings
Ncells = 20000
l1 = 0.12 # 0.12
l2 = -0.02 # -0.02
theta = 45
Kobs = 100 # observations T0
Lobs = 100 # observations T1
T0 = 25
T1 = -40
NX = 1000

#verifyhistograms(Ncells,Kobs,Lobs,l1,l2,theta,T0,T1,NX)
#verifyestimator_noT(Ncells,Kobs,l1,l2,T0,NX)
verifyestimator_withTsingleloop(Ncells,Kobs,l1,l2,theta,T0,T1,NX)


