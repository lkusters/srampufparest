# BEGIN Functions for calculating the likelihood grid
def calculate_likelihoods(NX, K, dT, l1, l2,th):
    from scipy.special import comb
    from scipy.stats import norm
    import numpy as np

    xx = [i for i in np.linspace(0,1,NX)]
    yy= [norm.cdf(l1*norm.ppf(xi)+l2) for xi in xx]
    p0_p = np.diff(yy)
    p0_xi = [x+0.5/NX for x in xx[:-1]]
            
    Pk1k2 = [[0]*(K+1) for k in range(K+1)]
    for k1 in range(K+1):
        comb1 = comb(K,k1)
        for k2 in range(K+1):
            # outer integraal (p(x_1|..), given by p0_p,p0_xi)
            total = 0
            for (xi1,pdfxi1) in zip(p0_xi,p0_p):
                # inner integral
                yy = [norm.cdf( (th/dT) * (norm.ppf(xi)-norm.ppf(xi1)) )   for xi in xx]
                p1_p = np.diff(yy)
                p1_xi = [x+0.5/NX for x in xx[:-1]]
                int2 = sum([(xi2**k2)*((1-xi2)**(K-k2))*pdfxi2 for (xi2,pdfxi2) in zip(p1_xi,p1_p)])
                # end inner integraal
                total = total+ pdfxi1*int2*(xi1**k1)*((1-xi1)**(K-k1))
                # end outer integraal
            Pk1k2[k1][k2] = comb1*comb(K,k2)*total
    return Pk1k2

#def calculate_likelihoods_par(th):
    # this function is used only to parallelize
#    global NX, K, dT, l1, l2,p0_p,p0_xi,xx # K
#    print(K)
#    return calculate_likelihoods(NX, K, dT, l1, l2,th,p0_p,p0_xi,xx)

def loop_calculate_likelihoods(NX, K, dT, Lambdas1, Lambdas2,Thetas):
    # now Lambdas1,Lambdas2,Thetas must be lists
    # REQUIRED: conda: ipcluster start -n 4
    from ipyparallel import Client
    import numpy as np
    from scipy.stats import norm
    import os

   
    rc = Client()
    print('activated workers')
    print(rc.ids)
    with rc[:].sync_imports():
        import os
        #from MyFunctions import calculate_likelihoods_par
    
    dv = rc[:]
    #%px cd workdir
    workdir = os.getcwd()
    print(workdir)
    dv.apply_sync(os.chdir,workdir )
    

    for l1 in Lambdas1:
        for l2 in Lambdas2:
            
            pr_list = dv.map_sync(calculate_likelihoods, [NX]*len(Thetas),[K]*len(Thetas),[dT]*len(Thetas),[l1]*len(Thetas),[l2]*len(Thetas),Thetas)
            # send information to the workers
            #print(dv.push(dict(NX=NX,K=K,l1=l1,l2=l2,dT=dT,p0_p=p0_p,p0_xi=p0_xi,xx=xx) ,block=True )) 
            #print('pushed it')
            #print(dv.pull('K', block=True))
            #pr_list = dv.map_sync(calculate_likelihoods_par, Thetas)
            #dv[0].apply(calculate_likelihoods_par,Thetas[0],{'NX':NX,'K':K,'l1':l1,'l2':l2})#,dT=dT,p0_p=p0_p,p0_xi=p0_xi,xx=xx))

            for Pk1k2,th in zip(pr_list,Thetas):
                np.savetxt('Pk1k2_K%d_dT%d_%02d_%02d_%02d_n%d.txt' %(K,dT,1000*l1,1000*l2,th,NX) , Pk1k2, delimiter=' ', newline='\n', header='Likelihood Pr_{k_1,k_2}(k_1,k_2 |dT,l1,l2,th), with K=%d,dT=%d,NX=%d l1=%0.5f,l2=%0.5f,theta=%d'%(K,dT,NX, l1, l2, th), comments='# ')
# END Functions for calculating the likelihood grid
# BEGIN Functions for loading the data and data properties
def loadUnique(folderpath):
    # folderpath : path of folder that has the data
    # we assume 96 devices
    # we return Nobservations x Ncells
    
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
