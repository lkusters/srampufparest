# BEGIN Functions for calculating the likelihood grid
def calculate_likelihoods(NX, K, L, dT, l1, l2,th):
    p0_p, p0_xi = pdfp0(l1,l2,NX)
            
    Pk1k2 = [[0]*(K+1) for k in range(K+1)]
    for k1 in range(K+1):
        for k2 in range(K+1):
            # outer integraal (p(x_1|..), given by p0_p,p0_xi)
            total = 0
            for (xi1,pdfxi1) in zip(p0_xi,p0_p):
                # inner integral
                p1_p, p1_xi = pdfp1(xi1,dT,l1,l2,th,NX)
                int2 = sum([(xi2**k2)*((1-xi2)**(K-k2))*pdfxi2 for (xi2,pdfxi2) in zip(p1_xi,p1_p)])
                # end inner integraal
                total = total+ pdfxi1*int2*(xi1**k1)*((1-xi1)**(K-k1))
                # end outer integraal
            Pk1k2[k1][k2] = total
    return Pk1k2
def calculate_loglikelihood_temp(hist2D,bins1,bins2,dT,NX, l1, l2,theta):
    # calculate observation likelihood for two temperatures, given l1,l2, theta
    # accuracy NX
    # input is 2D histogram of observed ones
    # output is loglikelihoods
    import numpy as np
    
    K = max(bins1)
    L = max(bins2)
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    
    logll = 0
    for (k,hist) in zip(bins1,hist2D):
        for (l,count) in zip(bins2,hist):
            int1 = 0
            for (p1,xi1) in zip(p0_p, p0_xi):
                (p1_p, p1_xi) = pdfp1(xi1,dT,l1,l2,theta,NX)
                int2 = sum([(xi2**l)*((1-xi2)**(L-l))*p2 for (p2,xi2)\
                            in zip(p1_p, p1_xi) ]) # inner integral
                int1 = int1 + p1*(xi1**k)*((1-xi1)**(K-k)) * int2

            logll = logll + count*np.log10( int1) 
    
    return logll

def loop_calculate_loglikelihoods_temp(hist2D,bins1,bins2,dT,NX, Lambdas1, Lambdas2, Thetas):
    # now Lambdas1,Lambdas2, Thetas must be lists
    # REQUIRED: conda: ipcluster start -n 4
    # calculate the likelihood of the observations given the lambdas and theta
    
    # accuracy NX
    # input is 2D histogram of observed ones
    # output is loglikelihoods
    
    dv = startworkers()
    
    LogLL = []
    for l1 in Lambdas1:
        logll = []
        for l2 in Lambdas2:
            pr_list = dv.map_sync(calculate_loglikelihood_temp, [hist2D]*len(Thetas), [bins1]*len(Thetas), [bins2]*len(Thetas), [dT]*len(Thetas), [NX]*len(Thetas), [l1]*len(Thetas), [l2]*len(Thetas),Thetas)
            logll.append(pr_list)
        LogLL.append(logll)
    print('Finished calculating log-likelihoods. Returning [L1 x L2 x Theta]'+\
          ', [%d x %d x %d]'%(len(Lambdas1),len(Lambdas2),len(Thetas)))
    return LogLL


def calculate_loglikelihood(hist,bins,NX, l1, l2):
    # calculate observation likelihood for one temperature, given l1,l2
    # accuracy NX
    # input is histogram of observed ones
    # output is loglikelihood
    import numpy as np
    
    K = max(bins)
    p0_p, p0_xi = pdfp0(l1,l2,NX)
    
    logll = sum([count*np.log10(sum([p0*(xi**k)*((1-xi)**(K-k)) for p0,xi \
                                     in zip(p0_p, p0_xi)])) \
            for (count,k) in zip(hist,bins)])
            
    return logll

def loop_calculate_loglikelihoods(hist,bins,NX, Lambdas1, Lambdas2):
    # now Lambdas1,Lambdas2 must be lists
    # REQUIRED: 
    # calculate the likelihood of the observations given the lambdas
    
    # accuracy NX
    # input is histogram of observed ones
    # output is loglikelihoods
    
    
    dv = startworkers()
    LogLL = []
    for l1 in Lambdas1:
        pr_list = dv.map_sync(calculate_loglikelihood, [hist]*len(Lambdas2), [bins]*len(Lambdas2), [NX]*len(Lambdas2), [l1]*len(Lambdas2),Lambdas2)
        LogLL.append(pr_list)
    print('Finished calculating log-likelihoods. Returning [L1 x L2] result'+\
          '[%d x %d]'%(len(Lambdas1),len(Lambdas2)))
    return LogLL

#def loop_calculate_likelihoods(NX, K, dT, Lambdas1, Lambdas2,Thetas):
#    # now Lambdas1,Lambdas2,Thetas must be lists
#    # REQUIRED: conda: ipcluster start -n 4
#    from ipyparallel import Client
#    import numpy as np
#    import os
#
#   
#    rc = Client()
#    print('activated workers')
#    print(rc.ids)
#    with rc[:].sync_imports():
#        import os
#        #from MyFunctions import calculate_likelihoods_par
#    
#    dv = rc[:]
#    #%px cd workdir
#    workdir = os.getcwd()
#    print(workdir)
#    dv.apply_sync(os.chdir,workdir )
#    
#
#    for l1 in Lambdas1:
#        for l2 in Lambdas2:
#            
#            pr_list = dv.map_sync(calculate_likelihoods, [NX]*len(Thetas),[K]*len(Thetas),[dT]*len(Thetas),[l1]*len(Thetas),[l2]*len(Thetas),Thetas)
#            # send information to the workers
#            #print(dv.push(dict(NX=NX,K=K,l1=l1,l2=l2,dT=dT,p0_p=p0_p,p0_xi=p0_xi,xx=xx) ,block=True )) 
#            #print('pushed it')
#            #print(dv.pull('K', block=True))
#            #pr_list = dv.map_sync(calculate_likelihoods_par, Thetas)
#            #dv[0].apply(calculate_likelihoods_par,Thetas[0],{'NX':NX,'K':K,'l1':l1,'l2':l2})#,dT=dT,p0_p=p0_p,p0_xi=p0_xi,xx=xx))
#
#            for Pk1k2,th in zip(pr_list,Thetas):
#                np.savetxt('Pk1k2_K%d_dT%d_%02d_%02d_%02d_n%d.txt' %(K,dT,1000*l1,1000*l2,th,NX) , Pk1k2, delimiter=' ', newline='\n', header='Likelihood Pr_{k_1,k_2}(k_1,k_2 |dT,l1,l2,th), with K=%d,dT=%d,NX=%d l1=%0.5f,l2=%0.5f,theta=%d'%(K,dT,NX, l1, l2, th), comments='# ')
## END Functions for calculating the likelihood grid
# BEGIN Functions for loading the data and data properties
def loadUnique(folderpath):
    # folderpath : path of folder that has the data
    # we assume 96 devices
    # OUT
    # observations : Ndevices x Nobservations x Ncells
    # mergedobservations : Ncells x Nobservations
    
    import os
    import numpy as np
    
    os.path.normpath(folderpath)
    files = os.listdir(folderpath)
    
    observations = []
    for device in range(96):
        # take only files in this device
        observations_dev = []
        for filename in files:
            if filename.startswith('Unique_dev%03d'%(device+1)):
                filepath = os.path.normpath(folderpath+ '\\' +filename)
                data =  np.fromfile(filepath, np.uint8)
                data_bits = np.unpackbits(data)
                observations_dev.append(data_bits)
        observations.append(observations_dev)
    (Ndevices,Nobservations,Ncells) = np.shape(observations)
    
    mergedobservations = np.reshape(np.swapaxes(observations,1,2),\
                                    (Ncells*Ndevices,Nobservations))
    
    print('Finished loading data for %d devices, '%Ndevices+\
          'with %d observations of '%Nobservations+\
          '%d cells'%Ncells )
    print('returning observations [%d,%d,%d] '%np.shape(observations)+\
          'and merged observations [%d,%d]'%np.shape(mergedobservations))
    return observations,mergedobservations
# END Functions for loading the data and data properties

def getcounts1D(observations):
    # observations : Ncells x Nobservations
    # calculate for each cell the number of ones
    # then return the histogram of ones
    
    K = len(observations[0]) # number of observations per cell
    bins = [i for i in range(K+1)]
    bin_edges = [i-0.5 for i in range(K+2) ]
    import numpy as np
    hist, _ = np.histogram([sum(counts) for counts in observations],bins=bin_edges)
    print('Finished generating histogram, ' +\
          'with %d max observations '%K+\
          'and %d total cells'%sum(hist) )
    return hist, bins
    
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
    x_edges = [i for i in range(K1+2)]
    y_edges = [i for i in range(K2+2)]
    
    hist, _, _ = np.histogram2d([sum(counts) for counts in \
                                           observations1],\
        [sum(counts) for counts in observations2],\
        bins = [x_edges, y_edges])
    print('Finished generating 2D histogram, ' +\
          'with %d max observations K and %d max observations L '%(K1,K2) )
    return hist, bins1, bins2

def pdfp0(l1,l2,NX):
    # p0 as defined in SRAM-PUF model documentation
    # parameters lambda1,lambda2 accuracy NX (approx stepsize 1/NX)
    import numpy as np
    from scipy.stats import norm
    
    xx = [i for i in np.linspace(0,1,NX)]
    yy= [norm.cdf(l1*norm.ppf(xi)+l2) for xi in xx]
    p0_p = np.diff(yy)
    p0_xi = [x+0.5/NX for x in xx[:-1]]
    
    return p0_p, p0_xi

def pdfp1(xi1,dT,l1,l2,th,NX):
    # p1(xi2|xi1,dT,..) as defined in SRAM-PUF model documentation
    # parameters lambda1,lambda2, theta accuracy NX (approx stepsize 1/NX)
    # dT is temperature difference
    from scipy.stats import norm
    import numpy as np
    
    xx = [i for i in np.linspace(0,1,NX)]
    yy = [norm.cdf( (th/dT) * (norm.ppf(xi)-norm.ppf(xi1)) )   for xi in xx]
    p1_p = np.diff(yy)
    p1_xi = [x+0.5/NX for x in xx[:-1]]
    
    return p1_p, p1_xi

def startworkers():
    # requires conda: ipcluster start -n 4
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
    
    return dv