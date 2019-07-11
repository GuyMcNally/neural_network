from numpy import array, testing, dot
from neural_network.models.neural_network import NeuralNetwork

seed = 1

def test___sigmoid_derivative_1():
    neural_network = NeuralNetwork(seed)
    expected = 0.25
    result = neural_network.sigmoid_derivative(0.5)
    assert expected == result

def test___sigmoid_derivative_2():
    neural_network = NeuralNetwork(seed)
    expected = 0.1875
    result = neural_network.sigmoid_derivative(0.75)
    assert expected == result