# NodeJS and Python TensorFlow Integration
Integration between NodeJS and TensorFlow for fast REST APIs and great Neural Networks

# How to use

#### You need Python 3.6 and numpy 
Clone the repository
```
git clone https://github.com/gabrielfreire/NodeJS-TensorFlow-First-NeuralNetwork.git
```
Go to the directory
```
cd {dirname}
```
Run node
```
node index.js
```
The node script will run ``gadient_descent_nn.py`` to train the model and asynchronously call ``load_model.py`` after training

change ``var data`` variable to whatever data you want to pass as input/output

# Folders
Folder containing events file for Tensorboard visualization
```js
events/
``` 

Folder containing saved model for future loading 
```js
tmp/
```