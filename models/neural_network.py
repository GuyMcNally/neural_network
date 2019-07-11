from numpy import exp, array, random, dot
from .neuron import Neuron

class NeuralNetwork:

  def __init__(self, seed):
    self.neuron = Neuron(3, 1, seed)
    self.neurons = [Neuron(3, 1, seed), Neuron(3, 1, seed)]

  # The derivative of the Sigmoid function.
  # This is the gradient of the Sigmoid curve.
  # It indicates how confident we are about the existing weight.
  def sigmoid_derivative(self, x):
      return x * (1 - x)

  def get_error_rate(self, training_set_outputs, predicted_output):
    # Calculate the error (The difference between the desired output and the predicted output).
    return training_set_outputs - predicted_output

  def get_adjustment_rate(self, training_set_inputs, error, predicted_output):
      # Multiply the error by the input and again by the gradient of the Sigmoid curve.
      # This means less confident weights are adjusted more.
      # This means inputs, which are zero, do not cause changes to the weights.
      sigmoid_derivative_curve = error * self.sigmoid_derivative(predicted_output)
      training_set_inputs_shape = training_set_inputs.T
      return dot(training_set_inputs_shape, sigmoid_derivative_curve)

  def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
      # Pass the training set through our neural network (a single neuron).
      predicted_output = self.neuron.think(training_set_inputs)

      error = self.get_error_rate(training_set_outputs, predicted_output)
      adjustment = self.get_adjustment_rate(training_set_inputs, error, predicted_output)

      # Adjust the weights.
      self.neuron.adjust_weights(adjustment)

  def trains(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for neuron in self.neurons:
      # Pass the training set through our neural network (a single neuron).
      predicted_output = neuron.think(training_set_inputs)
      error = self.get_error_rate(training_set_outputs, predicted_output)
      adjustment = self.get_adjustment_rate(training_set_inputs, error, predicted_output)

      # Adjust the weights.
      neuron.adjust_weights(adjustment)
