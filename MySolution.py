import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import cvxpy as cp

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm

        # set seed
        np.random.seed(0)

        # number of additional nonlinear functions
        self.EXTRA_FUNCTIONS = 5000

        # extra function parameter values
        self.R = None
        self.s = None

        # used to store the K "class K or not" classifiers
        self.W = None

        self.reg = 0.1 # regularization term

    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        # m number of n-dimensional samples
        m, n = trainX.shape

        self.R = np.random.randn(self.EXTRA_FUNCTIONS, n).astype(np.float64) # normal vectors
        self.s = np.random.randn(self.EXTRA_FUNCTIONS).astype(np.float64) # biases

        extra_func_values = (self.R @ trainX.T).T + self.s # m x EXTRA_FUNCTIONS
        activations = np.maximum(extra_func_values, 0) # apply relu
        A = np.hstack((np.ones(m).reshape(-1,1), activations, trainX)) # m x (1 + EXTRA_FUNCTIONS + n)

        # we will minimize the L1 norm of Ax - label_i , where label_i is 1 if == i and -1 else

        self.W = np.zeros((self.K, A.shape[1]))

        for i in range(self.K):
            b = np.where(trainY == i, 1, -1)
            ones_m = np.ones(m)
            ones_n = np.ones(A.shape[1])

            x = cp.Variable(A.shape[1])
            t = cp.Variable(A.shape[0])

            z = cp.Variable(A.shape[1])

            prob = cp.Problem(cp.Minimize(t @ ones_m + self.reg * (z @ ones_n)), [((A@x) - b) <= t, -1*((A@x)-b) <= t, x <= z, -x <= z])
            prob.solve()

            self.W[i,:] = np.array(x.value)

    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''

        m, n = testX.shape

        activations = np.maximum((self.R @ testX.T).T + self.s, 0) # m x EXTRA_FUNCTIONS
        ones = np.ones(m).reshape(-1,1) # m x 1

        A = np.hstack((np.ones(m).reshape(-1,1), activations, testX)) # m x (1 + EXTRA_FUNCTIONS + n)

        predict = A @ self.W.T # m x k
        predY = np.argmax(predict, axis=1) # m dimensional

        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None
        self.maxIter = 100

        np.random.seed(0)


    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''

        m, n = trainX.shape

        self.labels = np.random.randint(0, self.K, size=m)
        
        ind = np.random.choice(m, size=self.K, replace=False)

        # self.Z = np.zeros((n, self.K)) # Column i of Z is the i-th representative
        self.Z = trainX[ind,:].T
        self.D = np.zeros((self.K, m))

        Jprev = np.nan

        for iter in range(self.maxIter): #iterations of the algorithm
            for i in range(self.K): # For each cluster
                I = np.where(self.labels == i)[0]

                active_points = trainX[I,:] # selected points x n
                if active_points.shape[0] == 0: # no active points
                    continue

                x = cp.Variable(n)
                T = cp.Variable((active_points.shape[0], active_points.shape[1]))
                prob = cp.Problem(cp.Minimize(cp.sum(T)), [(active_points - x) <= T, -(active_points - x) <= T])
                prob.solve()

                self.Z[:,i] = np.array(x.value)

            for i in range(self.K):
                rep = np.squeeze(self.Z[:,i]) # n 
                self.D[i,:] = np.sqrt(np.sum((trainX - rep)**2, axis=1))

            d = np.min(self.D, axis=0)
            self.labels = np.argmin(self.D, axis=0)


            J = (1 / n) * (np.linalg.norm(d)**2)

            if iter > 1:
                if np.abs(J - Jprev) < (1e-5 * J):
                    break
                Jprev = J


        # Update and return the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        # m samples of n-dim feature vectors

        m,n = testX.shape

        D = np.zeros((self.K, m))

        for i in range(self.K):
            rep = np.squeeze(self.Z[:,i]) # n 
            D[i,:] = np.sqrt(np.sum((testX - rep)**2, axis=1))

        pred_labels = np.argmin(D, axis=0)

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1].astype(np.int64)).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 (Option 1) ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    