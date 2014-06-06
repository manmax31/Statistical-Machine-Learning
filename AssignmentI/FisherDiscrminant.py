'''
COMP 4670/8600: Introduction to Machine Learning 2014
Question 2: Dimensionality Reduction
'''

import numpy as np
import matplotlib.pyplot as plt
 
def read_data():
    '''Reading Iris data from the file and separating them based on the 3 classes SETOSA, VERSICOLOR, VIRGINCA'''
    f = open("bezdekIris.data.txt", 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
   
    lines = [line.split(",") for line in lines if line] 
    
    class1 = np.array([line[:4] for line in lines if line[-1]=="Iris-setosa"], dtype=np.float)
    class2 = np.array([line[:4] for line in lines if line[-1]=="Iris-versicolor"], dtype=np.float)  
    class3 = np.array([line[:4] for line in lines if line[-1]=="Iris-virginica"], dtype=np.float)  
    
    return class1, class2, class3 

def mean_all(class1, class2, class3):
    ''' Calculating mean of each class and mean of the entire data'''
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    mean3 = np.mean(class3, axis=0)  
    mean = (len(class1)*mean1 + len(class2)*mean2 + len(class3)*mean3) / (len(class1)+len(class2)+len(class3))
    return mean1, mean2, mean3, mean

def calculate_Sw(class1, class2, class3, mean1, mean2, mean3):
    ''' Calculation of within class co-variance'''
    Sw = np.dot((class1-mean1).T, (class1-mean1)) + np.dot((class2-mean2).T, (class2-mean2)) + np.dot((class3-mean3).T, (class3-mean3))
    return np.matrix(Sw)
 
def calculate_Sb(class1, class2, class3, mean1, mean2, mean3, mean):
    ''' Calculation of between class co-variance'''
    Sb = len(class1)*np.dot(np.matrix(mean1-mean).T, np.matrix(mean1-mean)) + len(class2)*np.dot(np.matrix(mean2-mean).T, np.matrix(mean2-mean)) + len(class3)*np.dot(np.matrix(mean3-mean).T, np.matrix(mean3-mean))
    return Sb

def calculate_Eigens(Sw, Sb):
    ''' Calculation of Eigenvectors and eigenvalues from Within-Class co-varinace and Between-class co-variance'''
    evals, evecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    evals = np.matrix(evals)
    #normaliser = np.sqrt(np.dot(np.matrix(evals), np.matrix(evals).T))
    return evals, evecs#, normaliser
    
def calculate_J(evals): 
    ''' Calculation of Fisher Criterion.
    The trace of matrix is sum of its eigenvalues'''
    return np.sum(evals)

def calculate_w(evecs, evals):
    ''' Calculation of normalised w determined by D' eigenvectors with largest eigenvalues'''
    normaliser = np.sqrt(np.dot(np.matrix(evals), np.matrix(evals).T))
    return np.matrix(evecs/normaliser)

def print_J(evals):
    ''' Calculation of J for all possible 2D planes which 4C2=6'''
    j_max = calculate_J(evals[0:2])
    print('D\'=2, J _0&1 = ', round(j_max, 2))
    
    evals02 = np.matrix([evals.item(0), evals.item(2)])
    print('J _0&2 = ', round(calculate_J(evals02),2))
    
    evals03 = np.matrix([evals.item(0), evals.item(3)])
    print('J _0&3 = ', round(calculate_J(evals03),2))
    
    evals12 = np.matrix([evals.item(1), evals.item(2)])
    print('J _1&2 = ', round(calculate_J(evals12),2))
    
    evals13 = np.matrix([evals.item(1), evals.item(3)])
    print('J _1&3 = ', round(calculate_J(evals13),2))
    
    evals23 = np.matrix([evals.item(2), evals.item(3)])
    print('J _2&3 = ', round(calculate_J(evals23),2))

def plot_iris(w, class1, class2, class3):
    ''' Function to plot 4D data to 2D space'''
    class1=np.matrix(class1)
    class2=np.matrix(class2)
    class2=np.matrix(class2)
    plt.figure(0)
    plt.plot(np.dot(class1, w.T), [0]*class1.shape[0], "bo", label="Iris-setosa")
    plt.plot(np.dot(class2, w.T), [0]*class2.shape[0], "go", label="Iris-versicolor")
    plt.plot(np.dot(class3, w.T), [0]*class3.shape[0], "y+", label="Iris-virginica")
    plt.legend()
    plt.show()

def main():
    class1, class2, class3 = read_data()
    #print(class1)
    mean1, mean2, mean3, mean = mean_all(class1, class2, class3)
    #print(mean1)
    
    # Within Class Co-Variance
    Sw = calculate_Sw(class1, class2, class3, mean1, mean2, mean3)
    
    # Between Class Co-Variance
    Sb = calculate_Sb(class1, class2, class3, mean1, mean2, mean3, mean)
    
    # Calculation of Eigenvectors, Eigenvalues
    evals, evecs = calculate_Eigens(Sw, Sb)
    
    # Calculation of weight w
    w = calculate_w(evecs[0:2], evals)
    
    print('W: ', w,'\n')
    
    # Print J for all 2D planes
    print_J(evals)
   
    plot_iris(w, class1, class2, class3)
    
    
if __name__ == "__main__" : main()