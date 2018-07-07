import numpy as np
import LayerFunc as lf

class NN (object):
    def __init__(self, features, classes, hid_layers, reg, weight):
        self.features = features
        self.classes = classes
        self.hid_layers = hid_layers
        self.reg = reg
        self.weight = weight
        self.num_layers = 1 + len(hid_layers)
        self.param = {}
        
        dims = [features] + hid_layers + [classes]

        for i in range(self.num_layers):
            self.param['W' + str(i + 1)] = weight * np.random.randn(dims[i], dims[i + 1])
            self.param['b' + str(i + 1)] = np.zeros(dims[i + 1])

    def loss(self, X_inp, Y_inp = None):
        X_inp = np.reshape(X_inp,(X_inp.shape[0],-1))
        
        grad =  {}
        dx = {}
        cache_layer = {}
        layer = {}
        layer[0] = X_inp
        loss = 0.0


        for i in range(1, self.num_layers):
            layer[i], cache_layer[i] = lf.forward(layer[i - 1], self.param['W%d' % i], self.param['b%d' % i])

        WLast = 'W%d' % self.num_layers
        bLast = 'b%d' % self.num_layers

        scores, cache_scores = lf.a_forward(layer[self.num_layers - 1],self.param[WLast],self.param[bLast])
        
        if Y_inp is None:
            return scores
    
        loss, dscores = lf.softmax_loss(scores, Y_inp)

        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.param['W%d' % i]**2)

        dx[self.num_layers], grad[WLast], grad[bLast] = lf.a_backward(dscores, cache_scores)
        grad[WLast] += self.reg * self.param[WLast]

        for i in reversed(range(1, self.num_layers)):
            dx[i], grad['W%d' % i], grad['b%d' % i] = lf.backward(dx[i + 1], cache_layer[i])
            grad['W%d' % i] += self.reg * self.param['W%d' % i]

        return loss, grad

         