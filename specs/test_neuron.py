from numpy import array, testing, dot
from neural_network.models.neuron import Neuron

seed = 1

def test___get_synaptic_stating_weights_3_1():
    neuron = Neuron(3, 1, seed)
    expected = array([[-0.16595599],[ 0.44064899], [-0.99977125]])
    result = neuron.synaptic_weights
    testing.assert_allclose(expected, result)

def test___get_synaptic_stating_weights_4_2():
    neuron = Neuron(4, 2, seed)
    expected = array([[-0.16595599,  0.44064899], [-0.99977125, -0.39533485], [-0.70648822, -0.81532281], [-0.62747958, -0.30887855]])
    result = neuron.synaptic_weights
    testing.assert_allclose(expected, result)

def test_sigmoid_1():
    neuron = Neuron(1, 1, seed)
    expected = 0.7310585786300049
    result = neuron.sigmoid(1)
    assert expected == result

def test_sigmoid_2():
    neuron = Neuron(1, 1, seed)
    expected = 0.8807970779778823
    result = neuron.sigmoid(2)
    assert expected == result

def test_sigmoid_3():
    neuron = Neuron(1, 1, seed)
    expected = 0.9525741268224334
    result = neuron.sigmoid(3)
    assert expected == result

def test_convert_to_between_minus_one_and_one_1():
    neuron = Neuron(1, 1, seed)
    expected = 1
    result = neuron.convert_to_between_minus_one_and_one(1)
    assert expected == result

def test_convert_to_between_minus_one_and_one_2():
    neuron = Neuron(1, 1, seed)
    expected = 0.5
    result = neuron.convert_to_between_minus_one_and_one(0.75)
    assert expected == result

def test_convert_to_between_minus_one_and_one_3():
    neuron = Neuron(1, 1, seed)
    expected = -0.5
    result = neuron.convert_to_between_minus_one_and_one(0.25)
    assert expected == result

def test_think_1():
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    neuron = Neuron(3, 1, seed)
    expected = array([ [0.2689864], [0.3262757], [0.23762817], [0.36375058]])
    result = neuron.think(training_set_inputs)
    testing.assert_allclose(expected, result)

def test_think_2():
    training_set_inputs = array([[0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
    neuron = Neuron(4, 2, seed)
    expected = array([[0.33037528, 0.3067574 ], [0.13328558, 0.31647723], [0.29474597, 0.40741215], [0.15364951, 0.22958471]])
    result = neuron.think(training_set_inputs)
    testing.assert_allclose(expected, result)

# def test_adjust_weights():
#     training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
#     adjustment = dot(training_set_inputs.T, 0.028125)
#     neuron = Neuron(3, 1, seed)
#     # old_weights = neuron.synaptic_weights
#     expected = array([[4.834044]])
#     print(">>>>>>>>>>>>.adjustment: ", adjustment)
#     neuron.adjust_weights(adjustment)
#     result = neuron.synaptic_weights
#     print(">>>>>>>>>>>result: ", result)
#     testing.assert_allclose(expected, result)
#     # assert old_weights != result
