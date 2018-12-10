# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:48:28 2018

Here are all functions for generating and estimating parameters of the srampuf
model.

required packages: numpy, scipy, ipyparallel
@author: lkusters
"""

# -----------------------------------------------------------------------
#   Calculate pdf
# -----------------------------------------------------------------------
def pdfp0(l1,l2,NX):
    # p0 as defined in SRAM-PUF model documentation
    # parameters lambda1,lambda2 accuracy NX (approx stepsize 1/NX)
    import numpy as np
    from scipy.stats import norm
    
    xi = [i for i in np.linspace(0,1,NX)]
    p0_p = np.diff( norm.cdf(l1*norm.ppf(xi)+l2) )
    p0_xi = xi[:-1]+np.diff(xi)/2
    
    return p0_p, p0_xi
  
def pdfp1(xi1,dT,l1,l2,th,NX):
    # p1(xi2|xi1,dT,..) as defined in SRAM-PUF model documentation
    # parameters lambda1,lambda2, theta accuracy NX (approx stepsize 1/NX)
    # dT is temperature difference
    from scipy.stats import norm
    import numpy as np
    
    xi = [i for i in np.linspace(0,1,NX)]
    xi1 = [xi1]*len(xi)
    yy = norm.cdf( (th/dT) * (norm.ppf(xi)-norm.ppf(xi1)) )  
    p1_p = np.diff(yy)
    p1_xi = xi[:-1]+np.diff(xi)/2
    
    return p1_p, p1_xi

# -----------------------------------------------------------------------
#   Calculate Log likelihoods
# -----------------------------------------------------------------------
    
def loglikelihood_temperature(hist2D,binscK,binscL,dT,NX, l1, l2,theta):
    # calculate observations likelihood for two temperatures, 
    # given l1,l2, theta. Accuracy of pdf estimation is NX
    # input is 2D histogram of observed ones
    # output is loglikelihood
    # pdfp1 is slower than power calculation so just do each pdfp1 once!
    import numpy as np
    
    Kobs = max(binscK)
    Lobs = max(binscL)
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    binscK = np.asarray(binscK)
    binscL = np.asarray(binscL)
    
    TOTAL = [[0 for i in range(Lobs+1)] for j in range(Kobs+1)]
    for pxi0,xi0 in zip(p0_p, p0_xi):
        p1_p, p1_xi = pdfp1(xi0,dT,l1,l2,theta,NX)
        pKones = xi0**binscK*(1-xi0)**(Kobs-binscK)*pxi0
        pLones = [sum(p1_xi**l*(1-p1_xi)**(Lobs-l)*p1_p) for l in binscL]
        TOTAL += np.transpose(np.matlib.repmat(pKones,Lobs+1,1))*\
        np.matlib.repmat(pLones,Kobs+1,1)
    
    logll = sum(sum(np.log10(TOTAL)*np.asarray(hist2D)))
    return logll


def loglikelihood(hist,bins,NX, l1, l2):
    # calculate observation likelihood for one temperature, given l1,l2
    # Accuracy of pdf estimation is NX
    # input is histogram of observed ones
    # output is loglikelihood
    # note the strong relation with generateSRAMPUFkonesdistribution
    import numpy as np
    
    K = max(bins)
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    
    logll = sum([count*np.log10(sum(p0_p*(p0_xi**k)*((1-p0_xi)**(K-k)) )) \
                 for (count,k) in zip(hist,bins)])
            
    return logll


# -----------------------------------------------------------------------
#   Parallel processing
# -----------------------------------------------------------------------
    
def startworkers():
    # requires conda: ipcluster start -n 4
    global srampuf_DV
    if 'srampuf_DV' in globals():
        print('workers have already been activated')
        return srampuf_DV
    
    try:
        from ipyparallel import Client
   
        rc = Client()
        print('activated workers')
        print(rc.ids)
        with rc[:].sync_imports():
            from srampufparest import pdfp0
            from srampufparest import pdfp1
        
        dv = rc[:]
    except:
        print('failed to activate workers. Did you generate them ?')
        print('e.g. ipcluster start -n 4')
        raise
    
    srampuf_DV = dv
    return dv


def loop_loglikelihood(hist,bins,NX, Lambdas1, Lambdas2):
    # calculate observations likelihood for one temperature,
    # given lists with Lambdas1,Lambdas2.
    # Accuracy of pdf estimation is NX
    # input is histogram of observed ones
    # output is loglikelihoods
    
    LogLL = []
    for l1 in Lambdas1:
        LogLL.append([loglikelihood(hist,bins,NX, l1, l2) for l2 in Lambdas2])
    
    print('Finished calculating log-likelihoods. Returning [L1 x L2] result'+\
          '[%d x %d]'%(len(Lambdas1),len(Lambdas2)))
    return LogLL

def loop_loglikelihood_temperature_givenl1l2(hist2D,bins1,bins2,dT,NX, l1, l2, Thetas):
    # calculate observations likelihood for two temperatures, 
    # given lists with Lambdas1,Lambdas2, Thetas.
    # Accuracy of pdf estimation is NX
    # input is 2D histogram of observed ones
    # output is loglikelihoods
    # REQUIRED: conda: ipcluster start -n 4
    
    dv = startworkers()

    LogLL = dv.map_sync(loglikelihood_temperature, [hist2D]*len(Thetas), [bins1]*len(Thetas), [bins2]*len(Thetas), [dT]*len(Thetas), [NX]*len(Thetas), [l1]*len(Thetas), [l2]*len(Thetas),Thetas)
    
    print('Finished calculating log-likelihoods. Returning [Theta ]'+\
          ', [%d ] result'%len(Thetas))
    return LogLL

# REMOVE THIS PARALLEL IMPLEMENTATION
#def loop_loglikelihood_temperature(hist2D,bins1,bins2,dT,NX, Lambdas1, Lambdas2, Thetas):
#    # calculate observations likelihood for two temperatures, 
#    # given lists with Lambdas1,Lambdas2, Thetas.
#    # Accuracy of pdf estimation is NX
#    # input is 2D histogram of observed ones
#    # output is loglikelihoods
#    # REQUIRED: conda: ipcluster start -n 4
#    
#    dv = startworkers()
#    
#    LogLL = []
#    for theta in Thetas:
#        print('theta = {0}'.format(theta))
#        logll = []
#        for l1 in Lambdas1:
#            pr_list = dv.map_sync(loglikelihood_temperature, [hist2D]*len(Lambdas2), [bins1]*len(Lambdas2), [bins2]*len(Lambdas2), [dT]*len(Lambdas2), [NX]*len(Lambdas2), [l1]*len(Lambdas2), Lambdas2,[theta]*len(Lambdas2))
#            logll.append(pr_list)
#        LogLL.append(logll)
#    print('Finished calculating log-likelihoods. Returning [Theta x L1 x L2]'+\
#          ', [%d x %d x %d]'%(len(Thetas),len(Lambdas1),len(Lambdas2)))
#    return LogLL

# -----------------------------------------------------------------------
#   Get the Histograms
# -----------------------------------------------------------------------
    
def getcounts1D(observations):
    # observations : Ncells x Nobservations
    # calculate for each cell the number of ones
    # then return the histogram of ones
    
    K = len(observations[0]) # number of observations per cell
    centerpoints = [i for i in range(K+1)]
    bin_edges = [i-0.5 for i in range(K+2) ]
    import numpy as np
    hist, _ = np.histogram([sum(counts) for counts in observations],bins=bin_edges)
    print('Finished generating histogram, ' +\
          'with %d max observations '%K+\
          'and %d total cells'%sum(hist) )
    return hist, centerpoints
    
def getcounts2D(observations1,observations2):
    # observations1 : Ncells x Nobservations
    # observations2 : Ncells x Nobservations
    # calculate for each cell the number of ones in set 1 and set 2
    # then return the histogram of ones

    import numpy as np
    K1 = len(observations1[0]) # number of observations per cell
    K2 = len(observations2[0]) # number of observations per cell
    bins1 = [i for i in range(K1+1)]
    bins2 = [i for i in range(K2+1)]
    x_edges = [i-0.5 for i in range(K1+2)]
    y_edges = [i-0.5 for i in range(K2+2)]
    
    hist, _, _ = np.histogram2d([sum(counts) for counts in \
                                           observations1],\
        [sum(counts) for counts in observations2],\
        bins = [x_edges, y_edges])
    print('Finished generating 2D histogram, ' +\
          'with %d max observations K and %d max observations L '%(K1,K2) )
    return hist, bins1, bins2

def getcellskones(observations,kones):
    # return cell index of all cells that have k ones
    idx = [i for i,counts in enumerate(observations) if sum(counts) == kones]
    return idx

  
# -----------------------------------------------------------------------
#   Store log-likelihoods / Load log-likelihoods
# -----------------------------------------------------------------------
def writeloglikelihoods(filename,LL,NX,Lambdas1,Lambdas2,Thetas):
    import numpy as np
    if Thetas == None: # no theta, 2D array
        np.savetxt(filename, LL, delimiter=' ', newline='\n', header=\
                   'log-likelihood of observed sequences as function of l1,'+\
                   'l2 , with\n'+\
                   'NX = %d\nl1 = %s\nl2 = %s'%(NX,' '.join(map(str, Lambdas1)),
                                                   ' '.join(map(str, Lambdas2))
                                                   ), comments='# ')
    else: # we have to store a 3D array
        with open(filename, 'w') as outfile:
            outfile.write(
                    '# log-likelihood of observed sequences as funct'+\
                    'ion of l1,l2 , with\n'+\
                    '# NX = %d\n# l1= %s\n# l2= %s\n# theta= %s\n'%(NX,
                    ' '.join(map(str, Lambdas1)),' '.join(map(str, Lambdas2)),
                    ' '.join(map(str, Thetas)) ) )
            count = 0
            for slice_2d in LL:
                outfile.write('# Theta = {0} \n'.format(Thetas[count]))
                np.savetxt(outfile, slice_2d)
                count+= 1;

def readloglikelihoods(filename):
    import numpy as np
    LL = NX = Lambdas1 = Lambdas2 = Thetas = None
    
    with open(filename, 'r') as infile:
        for i in range(10):
            line = infile.readline()
            if(line.startswith('# NX') ):
                NX = int(line.split('=')[1])
            elif(line.startswith('# l1') ):
                Lambdas1 = line.split('=')[1]
                Lambdas1 = [float(i) for i in Lambdas1.split(' ')[1::]]
            elif(line.startswith('# l2') ):
                Lambdas2 = line.split('=')[1]
                Lambdas2 = [float(i) for i in Lambdas2.split(' ')[1::]]
            elif(line.startswith('# theta') ):
                Thetas = line.split('=')[1]
                Thetas = [float(i) for i in Thetas.split(' ')[1::]]
            elif(not(line.startswith('#')) ):
                break
    LL = np.loadtxt(filename)
    if(Thetas != None):
        LL = LL.reshape(len(Thetas),len(Lambdas1),len(Lambdas2))
    
    return LL,NX,Lambdas1,Lambdas2,Thetas

# -----------------------------------------------------------------------
#   Generate synthetic data
# -----------------------------------------------------------------------
    
def generateSRAMPUFparameters(Ncells,l1,l2,theta):
    # generate the model parameters M,D for Ncells
    # l1 = sigma_N/Sigma_M
    # l2 = (t-mu_M)/sigma_M
    # theta = sigma_N/sigma_D
    # assume sigma_N = 1 , and t = 0, then
    import numpy as np
    sigmaM = 1/l1;
    muM = -l2/l1;
    sigmaD = 1/theta;
    
    M = np.random.normal(loc=muM,scale=sigmaM,size=[Ncells])
    D = np.random.normal(loc=0,scale=sigmaD,size=[Ncells])
    print('Finished generating cell-parameters for ' +\
          '%d cells'%(Ncells) )
    print('SETTINGS: mu_M = {0}, sigma_M = {1}'.format(muM,sigmaM))
    print('SETTINGS: mu_D = {0}, sigma_D = {1}'.format(0,sigmaD))
    return M,D

def generateSRAMPUFobservations(M,D,Nobs,temperature):
    # generate measurements for the SRAMPUFs with parameters M,D
    # We have Nobs at given temperature 
    import numpy as np
    sigmaN = 1
    
    samples = []

    for m,d in zip(M,D):
        samples.append(
                [1 if (i+m+d*temperature)>=0 else 0 for i in 
                 np.random.normal(loc=0,scale=sigmaN,
                                  size=Nobs)   ]   )
    Ncells = len(M)
    
    print('Finished generating %d cell-observations for %d cells'%(Nobs,Ncells) )
    print('SETTINGS: sigma_N = {0}'.format(sigmaN))
    return samples
   
def generateSRAMPUFkonesdistribution(Kobs,l1,l2,NX):
    # generate synthetic histogram
    from scipy.special import comb
    
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    binscK = [k for k in range(Kobs+1)]
    pKones = [comb(Kobs,k)*sum(p0_xi**k*(1-p0_xi)**(Kobs-k)*p0_p) for k in binscK]
    
    return pKones,binscK
    
def generateSRAMPUFklonesdistribution(Kobs,Lobs,dT,l1,l2,theta,NX):
    # generate synthetic histogram
    from scipy.special import comb
    import numpy as np
    
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    binscK = np.asarray([k for k in range(Kobs+1)])
    binscL = np.asarray([l for l in range(Lobs+1)])
    TOTAL = [[0 for i in range(Lobs+1)] for j in range(Kobs+1)]
    for pxi0,xi0 in zip(p0_p, p0_xi):
        p1_p, p1_xi = pdfp1(xi0,dT,l1,l2,theta,NX)
        pKones = comb(Kobs,binscK)*xi0**binscK*(1-xi0)**(Kobs-binscK)*pxi0
        pLones = [comb(Lobs,l)*sum(p1_xi**l*(1-p1_xi)**(Lobs-l)*p1_p) for l in binscL]
        TOTAL += np.transpose(np.matlib.repmat(pKones,Lobs+1,1))*\
        np.matlib.repmat(pLones,Kobs+1,1)
    return TOTAL,binscK,binscL


def generateSRAMPUFlonesdistribution(p1,Lobs,dT,l1,l2,theta,NX):
    # calculate the distribution of observing l ones of L observations
    # , given l1,l2, theta, dT (temperature difference), p1 (one-probability)
    # at the other temperature
    # Accuracy of pdf estimation is NX
    # output is pdf
    # this should match one-probability distribution at T=T1+dT of all cells 
    # that have one probability p1 at temperature T1
    from scipy.special import comb
    
    binscL = [l for l in range(Lobs+1)]
    p1_p, p1_xi = pdfp1(p1,dT,l1,l2,theta,NX)
    
    pLones = [comb(Lobs,l)*sum(p1_xi**l*(1-p1_xi)**(Lobs-l)*p1_p) for l in binscL]
            
    return pLones,binscL
