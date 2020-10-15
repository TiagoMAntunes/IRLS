import numpy as np
from parser import Parser

from scipy.special import expit

class IRLS:
    def result(v):
        return 1 if v >= 0.5 else 0

    def hypothesis(x,w): #sigmoid
        val = w.T.dot(x)
        return 1.0/(1+np.exp(-val))

    def test(filename, W):
        p = Parser(filename)
        X = p.transform_x()
        Y = p.transform_y()

        t_success = 0
        val_success = [0,0]
        count = [0,0]
        for i in range(len(X)):
            yi = Y[i]
            xi = X[i]
            res = result(hypothesis(xi, W))
            if res == yi:
                t_success += 1 
                val_success[yi] += 1
            count[yi] += 1
                
        print ('Correct: {}'.format(t_success * 100 / len(X)))
        print ('Positive correct {}'.format(val_success[1] * 100 / count[1]))
        print ('Negative correct {}'.format(val_success[0] * 100 / count[0]))

    np.set_printoptions(threshold=125)

    def train(filename):
        p = Parser(filename)
        X = p.transform_x().T # data points 
        Y = p.transform_y() # results

        W = np.zeros(124)
        
        #training
        for t in range(10):
            #no need to calculate R
            # ith row of X multiplies by Rii
            R = np.array([hypothesis(xi, W)*(1 - hypothesis(xi, W)) for xi in X.T])
            
            predictions = hypothesis(X, W)

            new_w = np.linalg.pinv((X*R).dot(X.T)).dot(X).dot(predictions - Y)
            
            W = W - new_w

        return W

model = train('a9a')


print('-------Test-------')
test('a9a.t', model)
print('-------Training-------')
test('a9a', model)

