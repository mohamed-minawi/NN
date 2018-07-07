import numpy as np
import LayerFunc as lf

class NN_Trainer(object):
    def __init__(self, model,data ,lr, batch_size, num_epochs, print_every, verbose, optim_config = None):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.optim_configs = optim_config
        self.lr_decay = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.print_every = print_every
        self.verbose = verbose

        self.reset()

    def reset(self):
       
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_loss = []
        self.val_acc_history = []
        
        self.optim_configs = {}
        for p in self.model.param:
            d = {k: v for k, v in self.optim_configs.items()}
            self.optim_configs[p] = d

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
          
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def step(self):
 
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.param.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = lf.adam(w, dw, config)
            self.model.param[p] = next_w
            self.optim_configs[p] = next_config
    
    def train(self):
       
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        for t in range(num_iterations):
            self.step()
            #if self.verbose:
            #    print '(Iteration %d / %d) loss: %f' % (
            #        t + 1, num_iterations, self.loss_history[-1])

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                loss, grads = self.model.loss(self.X_val, self.y_val)
                self.val_loss.append(loss)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print ("Epoch", self.epoch, "/", self.num_epochs, " train acc ", train_acc, " val acc", val_acc)

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.param.items():
                        self.best_params[k] = v.copy()

        self.model.param = self.best_params