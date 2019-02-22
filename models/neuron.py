from numpy import exp, array, random, dot

class Neuron:

  def __init__(self, num_input_connections, num_outputs_connections):
    self.synaptic_weights = self.__get_synaptic_stating_weights(num_input_connections, num_outputs_connections)


  def __set_up_seed(self, seed_number):
    random.seed(seed_number)

  def __get_synaptic_stating_weights(self, num_input_connections, num_outputs_connections):
    # Tells it to use the same random seed numbers each time
    self.__set_up_seed(1)

    # A single neuron
    # with x num_input_connections and x num_outputs_connections.
    # assigned random weights to a num_input_connections x num_outputs_connections matrix.
    random_nums_with_correct_shape = random.random((num_input_connections, num_outputs_connections))
    # Convert our current range of between [0,1] to [-1,1]
    # (2 * 1 = 2) - 1 = 1
    # (2 * 0 = 0) - 1 = 0
    # (2 * 0.25 = 0.5) - 1 = -0.5
    # This allows us to give give a neutral weight to 0.5 and negative/positive to anything on either side.
    return (2 * random_nums_with_correct_shape) - 1

  # The Sigmoid function, which describes an S shaped curve.
  # We pass the weighted sum of the inputs through this function to normalise them between 0 and 1.
  def sigmoid(self, x):
      return 1 / (1 + exp(-x))

  def think(self, inputs):
    # Pass inputs through our neural network (our single neuron).
    return self.sigmoid(dot(inputs, self.synaptic_weights))

  # We train the neural network through a process of trial and error.
  # Adjusting the synaptic weights each time.
  def train(self, training_set_inputs, training_set_outputs):
    # Pass the training set through our neural network (a single neuron).
    output = self.think(training_set_inputs)

    # Calculate the error (The difference between the desired output and the predicted output).
    error = training_set_outputs - output

    # Multiply the error by the input and again by the gradient of the Sigmoid curve.
    # This means less confident weights are adjusted more.
    # This means inputs, which are zero, do not cause changes to the weights.
    adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

    # Adjust the weights.
    self.synaptic_weights += adjustment
