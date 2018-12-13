# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:37:06 2018

@author: lkusters
"""

import srampufparest
import numpy as np
import matplotlib.pyplot as plt

folderpath = r'C:\Users\lkusters\surfdrive\python\temperature SRAM\data\SRAM\3'
#filepath = folderpath+r'\00_degrees_checked.txt_binary.txt'
#
#
#counts = []
#Ks = []
#for i in range(0,60,5):
#    filepath = folderpath+r'\%02d_degrees_checked.txt_binary.txt'%i
#    print(filepath)
#    data = srampufparest.loadData(filepath)
#    counts.append([sum(d) for d in data])
#    Ks.append(len(data[0]))
#
#temperature = np.asarray([i for i in range(0,60,5)])
#filename = folderpath+'\AllCounts_AllTemperatures.txt'    
#np.savetxt(filename, np.swapaxes(counts,1,2), fmt='%i', delimiter=' ', newline='\n', header=\
#                   'k-ones for each cell at different temperatures\n'+\
#                   'K = %s\nT = %s'%(' '.join(map(str, Ks)),' '.join(map(str, temperature))), comments='# ')
#                   
#counts = np.swapaxes(np.loadtxt(filename))
outputfilepath = folderpath+'\AllCounts_AllTemperatures.txt' 
srampufparest.loadAllDataStoreCounts(folderpath,outputfilepath)
allcounts, Ks, Ts = srampufparest.loadAllDataCounts(outputfilepath)

idx = 0
for c in allcounts:
    K = int(max(c))
    centerpoints = [i for i in range(K+1)]
    bin_edges = [i-0.5 for i in range(K+2) ]
    
    hist, _ = np.histogram(c,bins=bin_edges)
    plt.plot(centerpoints,hist)
    plt.title('T=%02d'%Ts[idx])
    plt.show()
    idx+=1
    
# conclude : 6th is weird (idx 5) so leave it out
counts = np.delete(allcounts,5,0)    
temperature = np.delete(Ts,5,0)

K = np.max(counts,axis=1)
for i in range(20,100):
    plt.plot(temperature,counts[:,i]/K)
    
plt.show()