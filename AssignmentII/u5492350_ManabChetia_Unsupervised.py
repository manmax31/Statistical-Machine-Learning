'''
Q 6b. Unsupervised Learning
Author: Manab Chetia, u5492350
'''

import numpy as np
import warnings
from scipy.stats import multivariate_normal

def read_data():
    ''' Reading IRIS Data and ignoring the class label'''
    f = open("bezdekIris.data.txt", 'r')
    return np.loadtxt(f, delimiter= ',', dtype="float", usecols=(0,1,2,3))

def intialize(K, data):
    '''Initialising mixing coefficient, mean and covariance to random values
    Input Parameters number of clusters K and Data
    Returns random mixing coefficients (1xK), mean(Kx4) and covariances (Kx4x4) arrays'''
    # Random initialisation of mixing coefficients
    mixing_coeff = np.random.dirichlet(np.ones(K), size=1)
    mixing_coeff = mixing_coeff[0]
    
    # Taking K random rows of the data set as mean for K clusters
    indexes = np.random.randint(data.shape[0], size=K)
    mean = data[indexes, :]
    
    # Creating a Kx(4x4) array for Covariance for K clusters. 
    C = np.cov(data.T) 
    covariance = np.repeat(C[None,:], K, axis=0) # Intialising the same covariance of the entire data to K clusters
    
    return mixing_coeff, mean, covariance

def expectation(K, data, mixcoeff, mean, covariance):
    '''Calculating responsibilities
    Input Parameters cluster K, data, mixing coefficients, mean and covariance
    Returns an array of responsibitlies'''
    responsibilities = np.zeros((data.shape[0], K)) # Storage for responsibilities
    for n in range(data.shape[0]):
        for k in range(K):
            numerator = mixcoeff[k] *  multivariate_normal.pdf(data[n], mean[k], covariance[k])
            denominator = getTotalGaussianValue(K, data[n], mixcoeff, mean, covariance) # Calculating the Normaliser
            responsibilities[n][k] = numerator/denominator
    return responsibilities

def maximisation(responsibilities, K, data): 
    '''Calulating the new mean, covariance and mixing coefficient
    Input parameters responsibilites, K, data'''  
    N_k = np.sum(responsibilities, axis=0)
    # Calculating the new mixing coefficients
    mixcoeff_new = np.array(N_k)/data.shape[0]
    # Calculating the new mean
    mean_new = (np.dot(data.T, responsibilities) / N_k).T
    # Calculating the new Covariance
    xbar = data - mean_new[:, None, :] 
    covariance_new = np.einsum('nk,knm,kno->kmo', responsibilities, xbar, xbar)
    covariance_new /= responsibilities.sum(axis=0)[:, None, None]
    
    return mixcoeff_new, mean_new, covariance_new
        
def getLogLikelihood(K, data, mixcoeff, mean, covariance):
    '''Calculating the Log Likelihood'''
    sum_n = 0.0
    for n in range(data.shape[0]):
        sum_k = 0.0
        for k in range(K):
            sum_k += mixcoeff[k] *  multivariate_normal.pdf(data[n], mean[k], covariance[k])
        sum_n += np.log(sum_k)
    return sum_n

def getExpectedLogLikelihood(K, data, mixcoeff, mean, covariance, responsibilities):
    '''Calculating the Expected Log Likelihood'''
    sum_n = 0.0
    for n in range(data.shape[0]):
        sum_k = 0.0
        for k in range(K):
            sum_k += responsibilities[n][k] * (np.log(mixcoeff[k]) +  np.log(multivariate_normal.pdf(data[n], mean[k], covariance[k])))
        sum_n += sum_k
    return sum_n
               
def getTotalGaussianValue(K, x, mixcoeff, mean, covariance):
    '''Calculating the Normaliser for the Expectation Step'''
    normaliser = 0.0
    for k in range(K):
        normaliser += mixcoeff[k] *  multivariate_normal.pdf(x, mean[k], covariance[k]) 
    return normaliser

def EM(K, data, start, threshold, maxIterations):
    '''Running the EM Algorithm'''
    np.random.shuffle(data) # Randomising the data
    mixcoeff, mean, covariance = intialize(K, data) # Intialising the values of pi, mu and sigma
    log_current = getLogLikelihood(K, data, mixcoeff, mean, covariance)
    #while(abs(log_current - log_new) > threshold):
    for i in range(maxIterations):
        responsibilities = expectation(K, data, mixcoeff, mean, covariance) # Calculating the responsibilities
        expectedLog = getExpectedLogLikelihood(K, data, mixcoeff, mean, covariance, responsibilities)
        mixcoeff, mean, covariance = maximisation(responsibilities, K, data) # Getting the new pi, mu and sigma
        log_new = getLogLikelihood(K, data, mixcoeff, mean, covariance) # Calculating the Log Likelihood
        if np.isnan(expectedLog) or np.isnan(log_current): # Breaking the loop if we get any Non Invertible arrays
            print('\nRestarting...')
            break
        #if(log_new <= log_current): # Checking for convergence
        if abs(log_new - log_current) < threshold: # Checking for convergence
            print(i+1, "Final Log: ", log_current, "Final Expected Log: ", expectedLog )
            print("Converged!")
            break
        print(i+1, "Log: ", log_current, "Expected Log: ", expectedLog )
        log_current = log_new
        
    
def main():
    warnings.filterwarnings("ignore") # Ignoring Warnings
    data = read_data() # Read Iris Data
    K = 3 # For 3 Clusters
    starts = 5 # Number of starts
    threshold = 0.1 # Setting a threshold for Convergence
    maxIterations = 150
    for start in range(starts):
        print( "Start:", start+1)
        EM(K, data, start, threshold, maxIterations)
        print()
        
if __name__ == '__main__':main()