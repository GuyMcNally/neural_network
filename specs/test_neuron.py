from numpy import array, testing
from neural_network.models.neuron import Neuron

def test___get_synaptic_stating_weights_3_1():
    neuron = Neuron(3, 1)
    expected = array([[-0.16595599],[ 0.44064899], [-0.99977125]])
    result = neuron.synaptic_weights
    testing.assert_allclose(expected, result)

def test___get_synaptic_stating_weights_4_2():
    neuron = Neuron(4, 2)
    expected = array([[-0.16595599,  0.44064899], [-0.99977125, -0.39533485], [-0.70648822, -0.81532281], [-0.62747958, -0.30887855]])
    result = neuron.synaptic_weights
    testing.assert_allclose(expected, result)

def test_sigmoid_1():
    neuron = Neuron(1, 1)
    expected = 0.7310585786300049
    result = neuron.sigmoid(1)
    assert expected == result

def test_sigmoid_2():
    neuron = Neuron(1, 1)
    expected = 0.8807970779778823
    result = neuron.sigmoid(2)
    assert expected == result

def test_sigmoid_3():
    neuron = Neuron(1, 1)
    expected = 0.9525741268224334
    result = neuron.sigmoid(3)
    assert expected == result