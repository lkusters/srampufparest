# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:58:56 2018

load data and convert to correct format

@author: lkusters
"""

  
# -----------------------------------------------------------------------
#   Load data, and generate histograms
# -----------------------------------------------------------------------
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

def loadData(filepath):
    # folderpath : path of folder that has the data
    # observations : Ncells x Nobservations
    import numpy as np
    
    with open(filepath) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    content = [np.fromstring(x, dtype=int, count=-1, sep=',') for x in content]
    content = np.swapaxes(content,0,1) # swap axes, s.t. 1st dim is cells, and second is observations
    
    return content