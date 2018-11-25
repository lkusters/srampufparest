from unittest import TestCase

import plot as plt
import numpy as np
import srampufparest

# if time write test cases here
# settings
NX = 100   # accuracy / number of samples

Lambdas1 = [i for i in np.linspace(0.1,0.13,10)] # found 0.1216
Lambdas2 = [i for i in np.linspace(-0.02,0.02,10)] # found -0.0005


obs1,merged1 = srampufparest.loadUnique(r'C:\Users\Lieneke\surfdrive\python\temperature SRAM\Unique\002_Temp_025')
hist,bins = srampufparest.getcounts1D(merged1)

test1 = srampufparest.calculate_loglikelihood(hist,bins,NX, 0.1, -0.02)

test2 = srampufparest.loop_calculate_loglikelihoods(hist,bins,NX, Lambdas1, Lambdas2)


fig, ax = plt.subplots()
im = ax.imshow(test2)
ax.set_xticks(np.arange(len(Lambdas2)))
ax.set_yticks(np.arange(len(Lambdas1)))
ax.set_xticklabels(Lambdas2)
ax.set_yticklabels(Lambdas1)
