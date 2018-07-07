import numpy as np

def a_forward(inp, wght , bias):
    activation = inp.reshape(inp.shape[0], wght.shape[0]).dot(wght) + bias
    mem = (inp, wght, bias)
    return activation, mem

def relu_f(out):
    output = np.maximum(0, out)
    mem = out
    return output , mem

def forward(inp, wght, bias):
    act, forward_cache = a_forward(inp,wght,bias)
    output, relu_cache = relu_f(act)
    cache = (forward_cache, relu_cache)
    return output, cache

def a_backward(dout, mem):
    inp, weight, bias = mem;
    dw = inp.reshape(inp.shape[0], weight.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(weight.T).reshape(inp.shape)
    return dx, dw, db

def relu_b(out, mem):
    dx = out
    dx[mem < 0] = 0
    return dx

def backward(dout, cache):
    forward_cache, relu_cache = cache
    da = relu_b(dout, relu_cache)
    dx, dw, db = a_backward(da, forward_cache)
    return dx, dw, db

def softmax_loss(x, y):
    
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def adam(x, dx, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)
    
    next_x = None

    learning_rate, beta1, beta2, eps, m, v, t \
        = config['learning_rate'], config['beta1'], config['beta2'], \
        config['epsilon'], config['m'], config['v'], config['t']

    t += 1
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx**2)

    mb = m / (1 - beta1**t)
    vb = v / (1 - beta2**t)

    next_x = -learning_rate * mb / (np.sqrt(vb) + eps) + x

    config['m'], config['v'], config['t'] = m, v, t
  
    return next_x, config
