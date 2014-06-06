'''
COMP 4670/8600: Introduction to Machine Learning 2014
Author:u5492350 Manab Chetia
Question 1.3: Laplace Approximation
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

z = np.linspace(0, 10, 100)

def calc_Normalisation(z, k):
    '''Calculate  Normaliser Z'''
    #part1 = z**k
    #part2 = np.exp(-z**2 / 2)
    #return np.dot(part1, part2.transpose()) * (10/100)
    return np.sum(z**k * np.exp(-z*z / 2)) * (10/100)

def p_z(z, k):
    ''' Calculate p(z)'''
    normalisation = calc_Normalisation(z, k)
    part1 = z**k
    part2 = np.exp(-z**2 / 2)
    return (1/normalisation) * part1 * part2

def mean(k):
    ''' Calculate Mean'''
    return np.sqrt(k)

def variance(k):
    ''' Calculate Variance'''
    return 1/( (k / (np.sqrt(k))**2) + 1)
    

plt.figure(0)
plt.plot(z, p_z(z, 0.5), z, mlab.normpdf(z, mean(0.5), variance(0.5)))
plt.title('k = 0.5')

plt.figure(1)
plt.plot(z, p_z(z, 3), z, mlab.normpdf(z, mean(3), variance(3)))
plt.title('k = 3')

plt.figure(2)
plt.plot(z, p_z(z, 5), z, mlab.normpdf(z, mean(5), variance(5)))
plt.title('k = 5')

plt.show()