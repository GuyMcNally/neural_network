from numpy import exp, array, random, dot
from .neuron import Neuron

class NeuralNetwork:

  def __init__(self, seed):
    self.neuron = Neuron(3, 1, seed)

  # The derivative of the Sigmoid function.
  # This is the gradient of the Sigmoid curve.
  # It indicates how confident we are about the existing weight.
  def sigmoid_derivative(self, x):
      return x * (1 - x)

  def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
      # Pass the training set through our neural network (a single neuron).
      output = self.neuron.think(training_set_inputs)

      # Calculate the error (The difference between the desired output and the predicted output).
      error = training_set_outputs - output

      # Multiply the error by the input and again by the gradient of the Sigmoid curve.
      # This means less confident weights are adjusted more.
      # This means inputs, which are zero, do not cause changes to the weights.
      adjustment = dot(training_set_inputs.T, error * self.sigmoid_derivative(output))

      # Adjust the weights.
      self.neuron.adjust_weights(adjustment)