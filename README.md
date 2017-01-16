# ISI_prediction_nn
A neural network that can be trained to predict the next interspike interval (ISI) by learning from the interspike intervals immediately preceeding.

The code in ISI_nn.py takes a vector of ISIs from data/ISI.mat and uses it for both training and testing. The ISIs were previously calculated from a voltage trace of a cholinergic neuron in the medial septum-diagonal band of Broca. Users can edit the number of prediction ISIs used (input_nodes), as well as the number of different training set sizes to try (dataset_sizes), for the creation of learning curves, the proportion of the data to use for training and testing (train_pct), and the number of learning iterations.

The output is a single number, which is a prediction of the ISI time (in seconds). This is then compared against the real ISI time, plus or minus some permissable error (correct_threshold).
