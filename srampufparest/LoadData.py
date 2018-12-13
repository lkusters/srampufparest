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
        progressBar(device,96)
        observations_dev = []
        for filename in files:
            if filename.startswith('Unique_dev%03d'%(device+1)):
                filepath = os.path.normpath(folderpath+ '\\' +filename)
                data =  np.fromfile(filepath, np.uint8)
                data_bits = np.unpackbits(data)
                observations_dev.append(data_bits)
        observations.append(observations_dev)
    (Ndevices,Nobservations,Ncells) = np.shape(observations)
    progressBar(1,1)
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
    # load a single data file from Dan's dataset
    import numpy as np
    
    with open(filepath) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    content = [np.fromstring(x, dtype=int, count=-1, sep=',') for x in content]
    content = np.swapaxes(content,0,1) # swap axes, s.t. 1st dim is cells, and second is observations
    
    return content

def generateCountsFile(folderpath,outputfilepath):
    # load all of the files (for each temperature) and store the counts
    import numpy as np
    print('load all data and convert to counts\nIN:{0}\nOUT:{1}\n'.format(folderpath,outputfilepath))
    counts = []
    Ks = []
    temperature = np.asarray([i for i in range(0,55+1,5)])
    for i,temp in enumerate(temperature):
        filepath = folderpath+r'\%02d_degrees_checked.txt_binary.txt'%temp
        progressBar(i,len(temperature))
        data = loadData(filepath)
        counts.append([sum(d) for d in data])
        Ks.append(len(data[0]))
    progressBar(1,1)
    
    np.savetxt(outputfilepath, np.swapaxes(counts,0,1), fmt='%i', delimiter=' ', newline='\n', header=\
                   'k-ones for each cell at different temperatures\n'+\
                   'K = %s\nT = %s'%(' '.join(map(str, Ks)),' '.join(map(str, temperature))), comments='# ')
                   
def loadCountsFile(filepath):
    # load file with counts that was stored by generateCountsFile()
    import numpy as np
    with open(filepath, 'r') as infile:
        for i in range(10):
            line = infile.readline()
            if(line.startswith('# K =') ):
                KK = line.split('=')[1]
                KK = [int(i) for i in KK.split(' ')[1::]]
            elif(line.startswith('# T =') ):
                TT = line.split('=')[1]
                TT = [int(i) for i in TT.split(' ')[1::]]
            elif(not(line.startswith('#')) ):
                break
    allcounts = np.swapaxes(np.loadtxt(filepath),0,1)
    
    return allcounts, KK, TT

def progressBar(value, endvalue, bar_length=20):
    import sys

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rProgress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()