from numpy import array
from models.neural_network import NeuralNetwork

# #Intialise a single neuron neural network.
# seed = 1
# neural_network = NeuralNetwork(seed)

# print("Random starting synaptic weights: ")
# print(neural_network.neuron.synaptic_weights)

# # The training set. We have 4 examples, each consisting of 3 input values
# # and 1 output value.
# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# training_set_outputs = array([[0, 1, 1, 0]]).T

# # Train the neural network using a training set.
# # Do it 10,000 times and make small adjustments each time.
# neural_network.train(training_set_inputs, training_set_outputs, 10000)

# print("New synaptic weights after training: ")
# print(neural_network.neuron.synaptic_weights)

# # Test the neural network with a new situation.
# print("Considering new situation [1, 0, 0] -> ?: ")
# print(neural_network.neuron.think(array([1, 0, 0])))


#Intialise a two neuron neural network.
seed = 1
neural_network = NeuralNetwork(seed)

print("Random starting synaptic weights: ")
print(neural_network.neuron.synaptic_weights)

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T

# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
neural_network.trains(training_set_inputs, training_set_outputs, 10000)

print("New synaptic weights after training: ")
print("neuron 1: ", neural_network.neurons[0].synaptic_weights)
print("neuron 2: ", neural_network.neurons[1].synaptic_weights)

# Test the neural network with a new situation.
# print("Considering new situation [1, 0, 0] -> ?: ")
# print(neural_network.neuron.think(array([1, 0, 0])))