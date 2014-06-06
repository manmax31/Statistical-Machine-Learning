'''
Q6a. Supervised Learning
Author: Manab Chetia, u5492350
'''

import numpy as np
from scipy.stats import multivariate_normal

def read_data():
    '''Reading Iris data from the file and separating them based on the 3 classes SETOSA, VERSICOLOR, VIRGINCA'''
    f = open("bezdekIris.data.txt", 'r')
    #data = np.genfromtxt("bezdekIris.data.txt", delimiter=',', dtype=None, names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))
    lines = [line.strip() for line in f.readlines()]
    f.close()
    lines = [line.split(",") for line in lines if line] 
    data = np.array([line for line in lines])
    return data

def getMean(class1, class2, class3):
    ''' Calculating mean of each class'''
    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)
    mean3 = np.mean(class3, axis=0)  
    return mean1, mean2, mean3

def getCovariance(class1, class2, class3):
    ''' Calculating the Covariance of each class'''
    cov1 = np.cov(class1.T)
    cov2 = np.cov(class2.T)
    cov3 = np.cov(class3.T)
    return cov1, cov2, cov3

def getSFoldpairs(data, S):
    ''' Input Parameters: Data and S
    Generates Training and Test pairs based on fold number S and stores them in a dictionary with fold as keys and training and test data as values
    pairs = { fold1:[training_1, testing_1], fold2:[training_2, testing_2], ... }
    returns a dictionary  of all possible pairs '''
    pairs = {} 
    fold = 0
    for s in range(S):
        training = np.array([row for i, row in enumerate(data) if i % S != s])
        validation = np.array([row for i, row in enumerate(data) if i % S == s])
        pairs[fold] = [training, validation]
        fold += 1
    return pairs     

def getError(data, S):
    '''Calculating the Cross Validation Error'''
    np.random.shuffle(data) # Shuffling the data
    pairs = getSFoldpairs(data, 10); # Getting the S fold pairs
    error=0 # Keeping track of misclassifications
    count=0 # Keeping track of total samples encountered
    
    for fold in pairs.keys():
        traindata = pairs[fold][0] # Extract the training data from the values of the dictionary {fold:[[trainingdata] [testdata]]}
        testdata = pairs[fold][1] # Extracting the test data
        
        # Separating training data them into 3 classes
        class1 = np.array([item[:4] for item in traindata if item[4] == 'Iris-setosa'], dtype=np.float)
        class2 = np.array([item[:4] for item in traindata if item[4] == 'Iris-versicolor'], dtype=np.float)
        class3 = np.array([item[:4] for item in traindata if item[4] == 'Iris-virginica'], dtype=np.float)
    
        # Calculating the mean, covariance of all 3 classes
        mean1, mean2, mean3 = getMean(class1, class2, class3)
        cov1, cov2, cov3 = getCovariance(class1, class2, class3)
        
        # Iterating through each point of the test data
        for testPoint in testdata:
            testPointCord = np.array(testPoint[:4], dtype=np.float) # Getting the only the first 4 features of the testpoint
            testPointLabel = testPoint[4] # Getting the only label of the testpoint
            
            # Calculating p(x|Class)
            pClass1 = multivariate_normal.pdf(testPointCord, mean1, cov1)
            pClass2 = multivariate_normal.pdf(testPointCord, mean2, cov2)
            pClass3 = multivariate_normal.pdf(testPointCord, mean3, cov3)
        
            prob = {'Iris-setosa':pClass1, 'Iris-versicolor':pClass2, 'Iris-virginica':pClass3}
            predictedLabel = max(prob, key=prob.get) # Getting the label of the class with highest probability
            if(testPointLabel != predictedLabel): # Comparing predicted label with original label
                error += 1
            count += 1       
    print('Error:', str(round(error*100/count, 2))+'%')
   
def main():
    data = read_data() # Read the Data
    S = 10 # Number of Folds
    getError(data, S) # Getting the Cross Validation Error

if __name__ == "__main__" : main()