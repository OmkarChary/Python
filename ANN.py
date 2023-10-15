import numpy as np
from random import random
# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some predictions


class MLP(object):
  # A Multilayer Perceptron class.

  def __init__(self, num_inputs = 3, hidden_layers = [3, 3], num_outputs = 2):
    """ Constructor for the MLP. Takes teh number of inputs
         a variable number of hidden layers, and number of outputs

    Arg:
        num_inputs(int ): Number of inputs
        hidden_layers(list): A list of ints of the hidden layers
        num_outputs (int): Number of outputs
    """

    self.num_inputs = num_inputs
    self.hidden_layers = hidden_layers
    self.num_outputs = num_outputs

    # create a generic representation of the layers
    layers  = [num_inputs] + hidden_layers + [num_outputs]

    # create random connection weights for the layers
    weights = []
    for i in range(len(layers) - 1):
      w = np.random.rand(layers[i], layers[i + 1])
      weights.append(w)
    self.weights = weights
    #print('{}'.format(weights))

    derivatives = []
    for i in range(len(layers) - 1):
      d = np.zeros((layers[i], layers[i + 1]))
      derivatives.append(d)
    self.derivatives = derivatives
    #print('{}'.format(derivatives))

    activations = []
    for i in range(len(layers)):
      a = np.zeros(layers[i])
      activations.append(a)
    self.activations = activations
    #print('{}'.format(activations))



  def forward_propogate(self, inputs):
    """Computes forward propagation of the network based on input signals.

    Args:
        inputs(ndarray): Input signals
    Returns:
        activations (ndarray) : output values
    """

    # the input layer activation is just the input itself
    activations = inputs
    self.activations[0] = inputs

    # iterate through the network layers
    for i, w in enumerate(self.weights):

      # calculate matrix multiplication between previous activation and weight matrix
      net_inputs = np.dot(activations, w)

      # apply sigmoid activation function
      activations = self._sigmoid(net_inputs)

      # save the activations for backpropogation
      self.activations[i + 1] = activations

    # return output layer activation
    return activations

  def back_propogate(self, error, verbose = False):

    # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
    # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
    # s(h_[i+1]) = a_[i+1]

    # dE/dW_[i+1] = (y - a_[i+1]) s'(h_[i+1]) W_i s[(h_i)] a_[i-1]
    for i in reversed(range(len(self.derivatives))):
      #print(len(self.derivatives))
      activations = self.activations[i+1]
      delta = error * self._sigmoid_derivative(activations) # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
      #print('{}'.format(delta))
      delta_reshaped = delta.reshape(delta.shape[0], -1).T
      #print('{}'.format(delta_reshaped))
      current_activations = self.activations[i] # ndarray([0.1, 0.3]) --> ndarray([[0.1], [0.3])
      current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
      #print('{}'.format(current_activations_reshaped))
      self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
      #print('{}'.format(self.derivatives))
      error = np.dot(delta, self.weights[i].T)
    #print('{}'.format(self.derivatives))
      #if verbose:
        #print("Derivatives for W{} : {}".format(i, self.derivatives[i]))


    return error

  def gradient_descent(self, learning_rate):
    for i in range(len(self.weights)):

      weights = self.weights[i]
      #print('Original W{} {}'.format(i,weights))

      derivatives = self.derivatives[i]

      weights += derivatives * learning_rate
      #print('Updated W{} {}'.format(i,weights))

  def train(self, inputs, targets, epochs, learning_rate):

    for i in range(epochs):
      sum_error = 0
      for input, target in zip(inputs, targets):
        # forward propogation

        output = self.forward_propogate(input)
        #print('{}'.format(output))
        # calculate the error
        error = target - output

        # back propogation
        self.back_propogate(error)

        # apply gradient descent
        self.gradient_descent(learning_rate)

        # report error

        sum_error += self._mse(target, output)

      print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

  def _mse(self, target, output):
    return np.average((target - output)**2)




  def _sigmoid_derivative(self, x):
    return x * (1.0 - x)

  def _sigmoid(self, x):
    """ Sigmoid actvation fuction
    Args:
        x(float): Value to be processed
    Returns:
        y (float): Output
    """

    y = 1.0 / (1.0 + np.exp(-x))
    return y

if __name__ == "__main__":

  # create a dataset to train a network for the sum operation
  inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) # array ([[0.1, 0.2], [0.3, 0.4]])
  targets = np.array([[i[0] + i[1]] for i in inputs]) # array ([[0.3], [0.7]])

  # create an mlp
  mlp = MLP(2, [5], 1)

  # train our mlp
  mlp.train(inputs, targets, 50, 0.1)

  # create dummy data
  input = np.array([0.3, 0.4])
  target = np.array([0.7])

  output = mlp.forward_propogate(input)

  print()
  print()

  print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output))




