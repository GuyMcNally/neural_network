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

def test___get_error_rate():
    neural_network = NeuralNetwork(seed)
    training_set_outputs = array([[0, 1, 1, 0]]).T
    predicted_output = array([[0.2689864 ], [0.3262757 ], [0.23762817], [0.36375058]])
    expected = array([[-0.2689864 ], [0.6737243 ], [ 0.76237183], [-0.36375058]])
    result = neural_network.get_error_rate(training_set_outputs, predicted_output)
    testing.assert_allclose(expected, result)

def test___get_adjustment_rate():
    neural_network = NeuralNetwork(seed)
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    predicted_output = array([[0.2689864 ], [0.3262757 ], [0.23762817], [0.36375058]])
    error = array([[-0.2689864 ], [0.6737243 ], [ 0.76237183], [-0.36375058]])
    expected = array([[0.28621005], [0.06391297], [0.14913351]])
    result = neural_network.get_adjustment_rate(training_set_inputs, error, predicted_output)
    testing.assert_allclose(expected, result)

def test___train():
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    neural_network = NeuralNetwork(seed)
    expected = array([[ 0.12025406], [ 0.50456196], [-0.85063774]])
    neural_network.train(training_set_inputs, training_set_outputs, 1)
    result = neural_network.neuron.synaptic_weights
    testing.assert_allclose(expected, result)
