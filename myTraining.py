import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle 

def data_split(data, ratio):
    
    #To check the random values generated is correct or not
    np.random.seed(42)
    
    #It will produce random permutated/shuffled numbers
    shuffled = np.random.permutation(len(data))
    
    # Defining the size of test set
    test_set_size = int(len(data) * ratio)
    
    # test data rows
    test_indices = shuffled[:test_set_size]
    
    # Defining trainset size
    train_indices = shuffled[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

  
if __name__ == "__main__":
    # Loading and Reading  the data
    df = pd.read_csv('data.csv')

    # Calling the data_split() by passing the data and ratio
    train, test = data_split(df, 0.25)

    X_train = train[['fever', 'bodyPain', 'age', 'runnynose', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodyPain', 'age', 'runnynose', 'diffBreath']].to_numpy()

    # Reshaping the values
    Y_train = train[['infectionProb']].to_numpy().reshape(4185, )
    Y_test = test[['infectionProb']].to_numpy().reshape(1394, )

    # Build LogisticRegression Classifier Model
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    file.close()

