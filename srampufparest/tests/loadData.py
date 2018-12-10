# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:37:06 2018

@author: lkusters
"""

import srampufparest
import numpy as np
import matplotlib.pyplot as plt

folderpath = r'C:\Users\lkusters\surfdrive\python\temperature SRAM\data\SRAM\1'
filepath = folderpath+r'\00_degrees_checked.txt_binary.txt'


#counts = []
#Ks = []
#for i in range(0,60,5):
#    filepath = folderpath+r'\%02d_degrees_checked.txt_binary.txt'%i
#    print(filepath)
#    data = srampufparest.loadData(filepath)
#    counts.append([sum(d) for d in data])
#    Ks.append(len(data[0]))

filename = 'counts_temperatures.txt'    
#np.savetxt(filename, counts, delimiter=' ', newline='\n', header=\
#                   'k-ones for each cell at different temperatures'+\
#                   'K = %s'%(' '.join(map(str, Ks))), comments='# ')
                   
counts = np.loadtxt(filename)
temperature = np.asarray([i for i in range(0,60,5)])

idx = 0
for c in counts:
    K = int(max(c))
    centerpoints = [i for i in range(K+1)]
    bin_edges = [i-0.5 for i in range(K+2) ]
    
    hist, _ = np.histogram(c,bins=bin_edges)
    plt.plot(centerpoints,hist)
    plt.title('T=%02d'%temperature[idx])
    plt.show()
    idx+=1
    
# conclude : 6th is weird (idx 5) so leave it out
counts = np.delete(counts,5,0)    
temperature = np.delete(temperature,5,0)

K = np.max(counts,axis=1)
for i in range(20,100):
    plt.plot(temperature,counts[:,i]/K)
    
plt.show()