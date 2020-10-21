import numpy as np
from parser import Parser
from scipy.special import expit
from sklearn.model_selection import KFold
import scipy.sparse as sparse
from pprint import pprint
import pickle 
class RegularizedIRLS:
    def __init__(self, msteps=100):
        self.maxsteps = msteps
        self.models = []

    def hypothesis(self, x,w): #sigmoid
        val = w.T.dot(x)
        return 1.0/(1+np.exp(-val)) # if an np array is used it will return an array instead of a scalar

    def test(self, W, X, Y):
        """ Return the accuracy of the given weights over the given test """
        guesses = np.array(list(map(lambda x: 1 if x >= 0.5 else 0, self.hypothesis(X,W))))
        return np.mean(guesses == Y)

    def train(self, parserinput, regularization, threshold=5):
        
        def train_model(X,Y, regularization):
            """ Trains a model given input X, labels Y and with a given regularization """
            W = np.zeros(124)
            
            sparse_X = sparse.csr_matrix(X.T)

            #training
            steps = 0
            max_acc_count = 0
            best_accuracy = 0
            while max_acc_count < threshold and steps < self.maxsteps:
                steps+=1

                #no need to calculate total R
                # ith row of X multiplies by Rii
                miu = self.hypothesis(X,W) 
                R = np.array(miu*(1-miu))
                
                #common = (X * R) @ X.T
                common = sparse.csr_matrix(X*R).dot(sparse_X).toarray()
                
                predictions = miu

                hessian = np.linalg.pinv(regularization * np.identity(124) + common)
                new_W = hessian.dot(common.dot(W) + X.dot(Y - predictions))
                
                accuracy = self.test(W,X,Y)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    max_acc_count = 0
                else:
                    max_acc_count += 1

                W = new_W
            return steps, W


        def fold_train(X,Y,N, regularization):
            """ Applies training over N foldings"""

            kf = KFold(n_splits=N)
            models = []
            counter = 1
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                steps, W = train_model(X_train.T, Y_train, regularization)
                models.append((counter,steps,W))
                counter+=1

            return models
        
        p = parserinput
        X = p.transform_x() # data points 
        Y = p.transform_y() # labels
        
        steps, W = train_model(X.T,Y, regularization)
        print('[DEBUG] Trained without foldings')
        fold_res = ((0, steps, W), *fold_train(X,Y,2,regularization)) # 0 is always the non folded version
        print('[DEBUG] Folded training complete')
        self.models.append({'regularization': regularization, 'models': fold_res})
        return W

if __name__ == '__main__':
    m = RegularizedIRLS()
    
    pp = Parser('a9a')
    X = pp.transform_x()
    Y = pp.transform_y()
    for r in (0,0.001,0.003,0.01,0.03, 0.1,0.3,*range(1,100)):
        print('Starting train... r={}'.format(r))
        m.train(pp, r)
        print('Training finished')
    
    p = Parser('a9a.t')
    X_test = p.transform_x()
    Y_test = p.transform_y()

    
    accuracies = [] # training, test
    for model in m.models:
        tmp = []
        for x,y in ((X.T,Y), (X_test.T, Y_test)):
            non_folded_test = m.test(model['models'][0][2], x,y)
            vals = [m.test(w[2],x,y) for w in model['models']] # for each folding weights calculate the accuracy on both the training and test set
            tmp.append((non_folded_test,np.mean(vals)))
        accuracies.append(tmp)
            
    
    resobj = {'acc': accuracies, 'models': m.models}
    pickle.dump(resobj, open('resultsobj', 'wb'))
    
