import numpy as np
from PIL import Image

class Logistic(object):
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.random.normal(0, 1, (1, dim)) 

  
    def evaluate(self,phi_n):
        return sig(np.dot(self.weights, phi_n.T))

   
    def train(self, phi, labels, max_iter = 1000, learn_rate = 0.01):
        N = len(labels) 
        dim = self.dim 
        it = 0
        while True:
            it += 1
            if it > max_iter:
                break
            
           
            grad_E = [0 for i in range(dim)]
            for n in xrange(N):
                y_n = self.evaluate(phi[n]) 
                grad_E += (y_n - labels[n]) * phi[n] 

            w_prev = self.weights
            self.weights = w_prev - (learn_rate * np.array(grad_E)) 
            converged = False
            for x in (self.weights - w_prev):
                for y in x:
                    if abs(y) < 0.0001:
                        converged = True
                        break
            if converged:
                print 'Gradient descent converged in ' + str(it) + ' iterations'
                print 'LOGISTIC REGRESSION training complete!' 
                break

   
    def predict(self,phi_n):
        return int(round(self.evaluate(phi_n)[0]))

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))
