import numpy as np
from parser import Parser
from scipy.special import expit


class RegularizedIRLS:
    def __init__(self, msteps=100):
        self.maxsteps = msteps
        self.models = []

    def result(self, v):
        return 1 if v >= 0.5 else 0

    def hypothesis(self, x,w): #sigmoid
        val = w.T.dot(x)
        return 1.0/(1+np.exp(-val))

    def test(self, filename, W):
        p = Parser(filename)
        X = p.transform_x()
        Y = p.transform_y()

        t_success = 0
        val_success = [0,0]
        count = [0,0]
        for i in range(len(X)):
            yi = Y[i]
            xi = X[i]
            res = self.result(self.hypothesis(xi, W))
            if res == yi:
                t_success += 1 
                val_success[yi] += 1
            count[yi] += 1
                
        # print ('Correct: {}'.format(t_success * 100 / len(X)))
        # print ('Positive correct {}'.format(val_success[1] * 100 / count[1]))
        # print ('Negative correct {}'.format(val_success[0] * 100 / count[0]))


    def train(self, filename, regularization, threshold=10**-5):
        p = Parser(filename)
        X = p.transform_x().T # data points 
        Y = p.transform_y() # results

        W = np.zeros(124)
        
        #training
        diff = [0, 1]
        steps = 0
        while np.abs(diff[1] - diff[0]) > threshold and steps < self.maxsteps:
            steps+=1
            #no need to calculate R
            # ith row of X multiplies by Rii
            R = np.array([self.hypothesis(xi, W)*(1 - self.hypothesis(xi, W)) for xi in X.T])
            
            predictions = self.hypothesis(X, W)
            
            hessian = np.linalg.pinv(regularization * np.identity(124) + (X * R).dot(X.T))
            new_W = hessian.dot((X*R).dot(X.T).dot(W) + X.dot(Y - predictions))
            
            diff[0] = diff[1]
            diff[1] = np.linalg.norm(W - (-regularization*W + X.dot(Y-predictions)))
            
            W = new_W

            
        self.models.append((len(self.models), W, steps))
        return W

# regularization = 0.001
# model = train('a9a', regularization)

if __name__ == '__main__':
    m = RegularizedIRLS()
    models = [m.train('a9a', r) for r in (1, 5, 10)]
    for model in models:
        m.test('a9a.t', model)
        m.test('a9a', model)

