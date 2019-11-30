# Training detail
Set C0 always be 1 for both training  
Optimizer = SGD  
self.save_criterion = 0  
self.load_weights = 2  

Using multiple training, only change 3 hyper-parameters(epoch,lr,weight_decay) below during training  
## For MNIST:(5 runs)
1. Set epoch to 20  
self.lr = 0.01  
self.weight_decay = 0.00001  

2. Set epoch to 30  
self.lr = 0.001  
self.weight_decay = 0.0001  

3. Set epoch to 40  
self.lr = 0.0001  
self.weight_decay = 0.001  

4. Set epoch to 50  
self.lr = 0.00001  
self.weight_decay = 0.01  

5. Set epoch to 60  
self.lr = 0.000001  
self.weight_decay = 0.1

Then we get the model.pt.58 with test accuracy of 97.720%

## For Fashion MNIST:(5 runs)
1. Set epoch to 20  
self.lr = 0.01  
self.weight_decay = 0.00001  

2. Set epoch to 35  
self.lr = 0.001  
self.weight_decay = 0.0001  

3. Set epoch to 45  
self.lr = 0.0001  
self.weight_decay = 0.001  

4. Set epoch to 60  
self.lr = 0.00001  
self.weight_decay = 0.01  

5. Set epoch to 70
self.lr = 0.000001  
self.weight_decay = 0.1  

Then we get the model.pt.67 with test accuracy of 87.170%
