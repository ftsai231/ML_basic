import numpy as np


pKeep = 0.8
num_hidden_layer = 10
node_in_layer = 10
weights = np.random.rand(num_hidden_layer, node_in_layer)
dropout_prob = np.random.rand(weights.shape[0], weights.shape[1]) < pKeep
dropout_weights = weights * dropout_prob
