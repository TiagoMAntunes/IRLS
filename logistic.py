import numpy as np
from parser import Parser
import math

def result(v):
    return 1 if v >= 0.5 else 0

def hypothesis(x,w): #sigmoid
    val = w.T.dot(x)
    if val < 0:
        return 1 - 1 / (1 + math.exp(val))
    else:
        return 1/(1+math.exp(-val))


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
            
def train(filename, learning_rate, regularization):
    p = Parser(filename)
    X = p.transform_x() # data points 
    Y = p.transform_y() # results

    W = np.zeros(124)

    #training
    for t in range(1):
        new_w = np.copy(W)
        for i in range(len(X)): 
            xi = X[i]
            yi = Y[i]
            val = yi - hypothesis(xi, W)
            # if (yi == 0):
            #     print(yi, val, W.T.dot(xi))

            new_w += val * xi
        
        #new_w -= regularization * W

        #print(new_w - W)
        W += learning_rate*new_w

    return W

model = train('a9a', 0.03, 0.1)

"""print('-------Test-------')
test('a9a.t', model)
print('-------Training-------')
test('a9a', model)
"""
p = Parser('a9a')
X = p.transform_x()
W = model
a = 1.0 / (1 + np.exp(-np.dot(X,W)))