import numpy as np
from laminar import calc_jacobian
import adolc as ad
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1.0 - sigmoid(x))


class NeuralNetwork(object):
    def __init__(self, sizes = [2, 10, 1]):
        self.nlayer = len(sizes)
        self.sizes = sizes
        
        self.weights = []
        self.biases = []

        self.dweights = []
        self.dbiases = []

        self.nw = 0
        self.nb = 0
        for i in range(1, self.nlayer):
            weights = np.random.randn(sizes[i], sizes[i-1])
            bias = np.random.randn(sizes[i])

            self.nw += weights.size
            self.nb += bias.size
            self.weights.append(weights)
            self.biases.append(bias)
        self.n = self.nw + self.nb
    def eval(self, x):
        assert x.size == self.sizes[0]
        for i in range(1, self.nlayer):
            y = np.dot(self.weights[i-1], x) + self.biases[i-1]
            if i == self.nlayer-1:
                x = y
            else:
                x = sigmoid(y)
        return x[0]
    
    def set_from_vector(self, beta):
        assert beta.size == self.nw + self.nb
        start = 0
        for i in range(1, self.nlayer):
            end = start + self.weights[i-1].size
 #           print start, end
            self.weights[i-1] = np.reshape(beta[start:end], self.weights[i-1].shape)
            start = end

        for i in range(1, self.nlayer):
            end = start + self.biases[i-1].size
#            print start, end
            self.biases[i-1] = np.reshape(beta[start:end], self.biases[i-1].shape)
            start = end
  #      print end

    def dydbeta(self, x, beta):
        beta_c = beta.copy()
        beta = ad.adouble(beta)
        tag = 11
        ad.trace_on(tag)
        ad.independent(beta)
        self.set_from_vector(beta)
        y = self.eval(x)
        ad.dependent(y)
        ad.trace_off()
        beta = beta_c
        self.set_from_vector(beta_c)
        dJdbeta = calc_jacobian(beta, tag=tag, sparse=False)
        #print dJdbeta.shape
        return dJdbeta.reshape(beta_c.shape)
        
if __name__ == "__main__":
    def func(x):
        return 2.0*x*x + 10.0

    xd = np.random.randn(100)
    yd = func(xd)

    from pylab import *

    #figure()
    #plot(x, y, 'x')
    #show()
    
    nn = NeuralNetwork(sizes=[1, 4, 1])
    y = nn.eval(np.array([1.0]))
    print y
    beta = np.random.randn(nn.n)*0.01

    for j in range(5000):
        nn.set_from_vector(beta)
        dJdbeta = np.zeros_like(beta)
        J = 0.0
        for i in range(len(xd)):
            xin = np.array([xd[i]])
            yeval = nn.eval(xin)
            #print yeval
            J += (yeval - yd[i])**2
            dydbeta = nn.dydbeta(xin, beta)
            dJdbeta += 2.0*(yeval - yd[i])*dydbeta
            
        beta = beta - dJdbeta/np.abs(dJdbeta).max()*0.01
        if j%100 == 0:
            print j, J
        #print beta
    yeval = []
    for i in range(len(xd)):
        xin = np.array([xd[i]])
        yeval.append(nn.eval(xin))
    figure()
    plot(xd, yd, 'b.')
    plot(xd, yeval, 'rx')
    show()
        
