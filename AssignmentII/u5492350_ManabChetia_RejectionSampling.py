'''
Q 7: Rejection Sampling
Author: Manab Chetia, u5492350
'''

import numpy as np
import time as t
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def getOriginalDistribution(z):
    ''' Original Distribution '''
    p = lambda z : (0.3)*mlab.normpdf(z,5,0.5) + (0.3)*mlab.normpdf(z,9,2) + (0.4)*mlab.normpdf(z, 2, 20)
    plt.figure(0)
    plt.plot(z, p(z), label='p(z)')
    return p

def getProposalDistribution(mean, covariance, z, p):    
    ''' Proposal Distribution '''
    q = lambda z : mlab.normpdf(z, mean, covariance) # Getting the Proposal Distribution
    k = np.max(p(z)/q(z)) # Getting the Scaling Factor
    plt.plot(z, k*q(z), 'r', label = 'kq(z)')
    plt.legend()
    plt.xlabel('z')
    plt.title('Original and Proposal Distribution')
    return q, k
 
def getAcceptedSamples(mean, covariance, Nsamples, p, q, k): 
    '''Does Rejection Sampling, Returns 1000000 Accepted Samples
    Input parameters: mean and covariance of proposed distribution
    Number of accepted samples to be generated
    p: Original Distribution
    q: Proposed Distribution
    k: Scaling Factor
    Returns an array of Accepted Samples
    '''  
    Ngood = 0 # Current number of Accepted samples
    rejectedQty = 0 # Count of Rejected Samples
    startTime = t.time()
    goodSamples = np.zeros(Nsamples) # Storage for the accepted samples
    while Ngood < Nsamples :
        z0 = np.random.normal(mean, covariance, size=Nsamples) # Getting z0, as my Proposal Distribution is a Normal Distribution
        u0 = np.random.uniform(size=Nsamples) * k*q(z0) # Getting u0 a random uniform number
        ind, = np.where(p(z0) > u0) # Finds the indices where p(z0)>u0
        rejectedQty += Nsamples - ind.shape[0] # Calculates the rejected samples
        n = min(len(ind), Nsamples-Ngood)
        goodSamples[Ngood:Ngood+n] = z0[ind[:n]] # Accepts the goodSamples using the indices
        Ngood += n
    time1 = t.time() - startTime # Time required to execute 
    print('Rejected Samples:', rejectedQty)
    return goodSamples, time1

def createHistogram(goodSamples, minValue, maxValue, binWidth):
    '''Creates the Histogram from the Accepted Samples'''
    plt.figure(1)
    n, bins, patches = plt.hist(goodSamples, bins=np.arange(minValue, maxValue, binWidth), histtype='bar',normed=1)
    bins = bins[:len(bins)-1] + (bins[1]-bins[0])*0.5 # Taking the points in the bin centres
    plt.xlabel('z')
    plt.title('Histogram: Rejection Sampling')
    return n, bins

def sumSquaredErrors(p, n, bins):
    '''Histogram Approximation 
    Calculates and returns the Sum of Squared Errors'''
    pvalues = 0.1 * p(bins)
    n = 0.1 * n
    return  sum(pvalues-n)**2

def fasterImplementation(Nsamples, minValue, maxValue, binWidth):
    '''Generates 30% samples from N(5,0.5), 30% samples from N(9,2) and 40% samples from N(2,20)
    Creates the Histograms from the generated samples
    Return time required to get the all the samples'''
    startTime = t.time()
    z1 = np.random.normal(5, 0.5, size=0.3*Nsamples)
    z2 = np.random.normal(9, 2, size=0.3*Nsamples)
    z3 = np.random.normal(2, 20, size=0.4*Nsamples)
    samples = np.concatenate((z1, z2, z3))
    time2 = t.time() - startTime
    
    plt.figure(2)
    plt.hist(samples, bins=np.arange(minValue, maxValue, binWidth), histtype = 'bar', normed=1)
    plt.xlabel('z')
    plt.title('Histogram: Improved Sampling')
    return time2
   
def main():
    z = np.linspace(-50, 50, num=1000) # Getting Uniformly Spaces Zs
    # Mean and Covariance of my proposal distribution
    mean = 5
    covariance = 14.86
    print('Proposal Distribution: Mean:',mean,' Covariance:',covariance)
    # Number of Accepted Samples I want
    Nsamples = 1000000
    
    p = getOriginalDistribution(z) # Get the Original Distribution p(z)
    q, k = getProposalDistribution(mean, covariance, z, p) # Get Proposal Distribution q(z)
    goodSamples, t1 = getAcceptedSamples(mean, covariance, Nsamples, p, q, k) # Get all 1000000 Accepted Samples
    n, bins = createHistogram(goodSamples, minValue=-50, maxValue=50, binWidth=0.1) # Prints the Histogram and returns the values and bins of each bar
    print('Sum of Squared Errors: ', sumSquaredErrors(p, n, bins))
    t2 = fasterImplementation(Nsamples, minValue=-50, maxValue=50, binWidth=0.1)
    print('Time:  Rejection Sampling:',round(t1,3),'seconds     Improved Sampling:',round(t2,3),'seconds')
    plt.show()    
    
if __name__ == '__main__':main()