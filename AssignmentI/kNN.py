'''
COMP 4670/8600: Introduction to Machine Learning 2014
Question 3: CrossValidation and Classification
'''

from scipy.spatial import cKDTree as KDTree
import numpy as np


def read_data():
    '''Reading Iris data from the file and separating them based on the 3 classes SETOSA, VERSICOLOR, VIRGINCA
    returns the data associated with each class'''
    f = open("bezdekIris.data.txt", 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
   
    lines = [line.split(",") for line in lines if line] 
    
    class1 = np.array([line for line in lines if line[-1]=="Iris-setosa"])
    class2 = np.array([line for line in lines if line[-1]=="Iris-versicolor"])  
    class3 = np.array([line for line in lines if line[-1]=="Iris-virginica"])  
    
    return class1, class2, class3 


def misclassification(k, traindata, testdata):
    '''kNN Algorithm to calculate the misclassification error from k, training data and test data
    returns number of misclassifications found'''
    trainValues = np.array([line[:4] for line in traindata]) # Keeping only the features and removing the class labels from trainingdata
    
    tree = KDTree(trainValues) # Creating a kD Tree based on Training Data
    
    # Initialising variables for keeping track of Misclassifications
    error = 0
    
    for point in testdata: # Iterating through each sample of Training Data
        votes = {'Iris-setosa':0, 'Iris-versicolor':0, 'Iris-virginica':0 } # Dictionary to keep track of votes
        coordinate = np.array(point[:4]) # Extracting the 4 features (Sepal length, Sepal width, Petal Length, Petal Width) from a sample test point
        label = point[4] # Extracting the species label (Iris-setosa, Iris-versicolor, Iris-virginica) from the sample test point
        dist, indexes = tree.query(coordinate, k, p=2) # Getting the distance and indexes of the k Nearest Neighbours from our test point 'cordinate' # p=2 for euclidean distance
        if k==1:
            indexes = [indexes]
        else:
            indexes = indexes
        for index in indexes: # Iterating through the indexes
            votes[traindata[index][4]] += 1 # Counting votes 
         
        predictedLabel =  max(votes, key=votes.get) # Finding out which label got the maximum vote 
        
        if label != predictedLabel: # Checking if predicted label is equal to original label
            error += 1 # If predicted label and original label are not same, we count it as an error
    return error

def get_sfold_pairs(data, S):
    ''' Input Parameters: Data and S
    Generates Training and Test pairs based on S and stores them in a dictionary with index as keys and training and test data as values
    pairs = { 1:[training_1, testing_1], 2:[training_2, testing_2], ... }
    returns a dictionary  of all possible pairs '''
    pairs = {} 
    index = 0
    for s in range(S):
        training = np.array([row for i, row in enumerate(data) if i % S != s])
        validation = np.array([row for i, row in enumerate(data) if i % S == s])
        pairs[index] = [training, validation]
        index+=1
    return pairs

def crossValidation(k, S, traindata):
    '''Input parameters: k neighbors, S folds and trainingData
    returns the errors for all pairs of trainingData and testData returned by get_sfold_pairs(data, S) '''
    
    # Get all possible pairs of training data and test data for a particular value of S
    pairs = get_sfold_pairs(traindata, S) 
    
    errors = 0
    #    print('Traing:', pairs[0][0])
    for key in pairs.keys():
        traindata = pairs[key][0] # Extract the training data from the values of the dictionary {key:[trainingdata testdata]}
        testdata = pairs[key][1] # Extract the test data from the values of the dictionary {key:[trainingdata testdata]}
        error = misclassification(k, traindata, testdata)
        errors += error

    return errors/S #returning mean error 
    
def main():
    class1, class2, class3 = read_data() # Read data from text file and separate them based on class labels
    # Taking the first 80% data of each class as (training and cross validation data) and remaining 20% as test data
    traindata = np.concatenate((class1[:40,:], class2[:40,:], class3[:40,:])) # Size: 120
    testdata = np.concatenate((class1[40:,:], class2[40:,:], class3[40:,:])) # Size: 30
    
    S = 10 # Set Value of S fold here
    
    for k in range(1, 40, 2): # Generate k as 1,3,5,...,37,39
        print('S: ', S, '    k: ',k, '    Mean Error:', str(round(crossValidation(k, S, traindata), 2)))
  
    
if __name__ == "__main__" : main()
