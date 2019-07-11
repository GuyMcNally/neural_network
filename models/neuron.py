from numpy import exp, array, random, dot

class Neuron:

  def __init__(self, num_input_connections, num_outputs_connections, seed):
    self.synaptic_weights = self.__get_synaptic_stating_weights(num_input_connections, num_outputs_connections, seed)


  def __set_up_seed(self, seed_number):
    random.seed(seed_number)

  def convert_to_between_minus_one_and_one(self, number):
    # Convert our current range of between [0,1] to [-1,1]
    # (2 * 1 = 2) - 1 = 1
    # (2 * 1 = 2) - 1 = 1
    # (2 * 0 = 0) - 1 = -1
    # (2 * 0.25 = 0.5) - 1 = -0.5
    # This allows us to give a neutral weight to 0 and negative/positive to anything on either side.
    return (2 * number) - 1

  def __get_synaptic_stating_weights(self, num_input_connections, num_outputs_connections, seed):
    self.__set_up_seed(seed)
    # with x num_input_connections and x num_outputs_connections.
    # assigned random weights to a num_input_connections x num_outputs_connections matrix.
    random_nums_with_correct_shape = random.random((num_input_connections, num_outputs_connections))
    positive_or_negative_num =  self.convert_to_between_minus_one_and_one(random_nums_with_correct_shape)
    return positive_or_negative_num

  # We pass the weighted sum of the inputs through this function to normalise them between 0 and 1.
  def sigmoid(self, weights):
      exponentialNegativeWeight = exp(-weights)
      normalisedWeight = 1 / (1 + exponentialNegativeWeight)
      return normalisedWeight

  def think(self, inputs):
    # Pass inputs through our single neuron.
    product = dot(inputs, self.synaptic_weights)
    return self.sigmoid(product)

  def adjust_weights(self, adjustment):
    self.synaptic_weights += adjustment
