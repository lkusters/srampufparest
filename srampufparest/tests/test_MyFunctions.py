#from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import srampufparest
 
# if time write test cases here
# settings
NX = 10   # accuracy / number of samples

Lambdas1 = [i for i in np.linspace(0.1,0.13,4)] # found 0.1216
Lambdas2 = [i for i in np.linspace(-0.02,0.02,4)] # found -0.0005
Thetas = [i for i in np.linspace(10,80,4)] 

obs1,merged1 = srampufparest.loadUnique(r'C:\Users\Lieneke\surfdrive\python\temperature SRAM\Unique\002_Temp_025')
obs2,merged2 = srampufparest.loadUnique(r'C:\Users\Lieneke\surfdrive\python\temperature SRAM\Unique\001_Temp_m40')
dT = 65
hist2d,bins1,bins2 = srampufparest.getcounts2D(merged1,merged2)
hist1d,bins = srampufparest.getcounts1D(merged1)

#test1 = srampufparest.calculate_loglikelihood(hist1d,bins,NX, 0.1, -0.02)
ll1 = srampufparest.loop_calculate_loglikelihoods(hist1d,bins,NX, Lambdas1, Lambdas2)
np.savetxt('LL_T025.txt', ll1, delimiter=' ', newline='\n', header='log-likelihood of observed sequences as function of l1,l2 , with NX=%d.\n l1= %s \n l2= %s'%(NX,str(Lambdas1),str(Lambdas2)), comments='# ')

hist1d,bins = srampufparest.getcounts1D(merged2)
ll2 = srampufparest.loop_calculate_loglikelihoods(hist1d,bins,NX, Lambdas1, Lambdas2)
np.savetxt('LL_Tm40.txt', ll2, delimiter=' ', newline='\n', header='log-likelihood of observed sequences as function of l1,l2 , with NX=%d.\n l1= %s \n l2= %s'%(NX,str(Lambdas1),str(Lambdas2)), comments='# ')

#test3 = srampufparest.calculate_loglikelihood_temp(hist2d,bins1,bins2,dT,NX, 0.1, -0.02,10)
ll3 = srampufparest.loop_calculate_loglikelihoods_temp(hist2d,bins1,bins2,dT,NX, Lambdas1, Lambdas2, Thetas)
with open('LL_T025_Tm40.txt', 'w') as outfile:
    outfile.write('# log-likelihood of observed sequences as function of l1,l2 , with NX=%d.\n l1= %s \n l2= %s'%(NX,str(Lambdas1),str(Lambdas2)))
    outfile.write('# Array shape: {0}\n'.format(ll3.shape))
    for slice_2d in ll3:
        np.savetxt(outfile, slice_2d)
        outfile.write('# New slice\n')

# Read the array from disk
#new_data = np.loadtxt('test.txt')
# Note that this returned a 2D array!
# However, going back to 3D is easy if we know the 
# original shape of the array
#new_data = new_data.reshape((4,5,10))

#fig, ax = plt.subplots()
#im = ax.imshow(test2)
#ax.set_xticks(np.arange(len(Lambdas2)))
#ax.set_yticks(np.arange(len(Lambdas1)))
#ax.set_xticklabels(Lambdas2)
#ax.set_yticklabels(Lambdas1)
#ax.colorbar()
