# Quantum fox in a hole reinforcement learning model, by Aleksandar Kraljevic

The code consists of six .py files:
- quantum_model.py:
    This is the file that handles the training of the quantum reinforcement learning model, which in this case is equivalent to a DDQN, except that the DNN is replaced with a PQC.
- pqc.py:
    This file contains the code which initializes a PQC, and any other functions with relation to PQC. Think of functions that act as quantum gates, the rescaling of outputs, the handling of classical inputs, etc.
- evaluate_model.py:
    This code is for evaluating experiments that have already been performed. This includes the plotting of such experiments, as well as evaluating the performance of an already trained model over many samples.
- experiment.py:
    This makes use of the quantum_model.py and pqc.py in order to run larger experiments at once, such as hyperparameter tuning.
- fox_in_a_hole.py:
    This is the environment of the fox in a hole game. Other functions such as quantum_model.py and helper.py interact with this code in order to play the game and receive rewards.
- helper.py:
    This file contains a multitude of functions that are used by the other python files, in order to not clutter them. Think of functions whose job it is to plot data, compute results, compute exploration parameters, etc.

## Dependencies
The code uses some python packages that need to be installed to run:
- numpy
- matplotlib
- tensorflow (version 2.7.0)
- tensorflow_quantum
- cirq
- sympy
- seaborn
- pandas
- scipy

## Running the code
To run any of the python files, make sure to have all the .py files in the same folder. In addition to this, make sure to create three empty folders called "data", "models", and "plots". As the names suggest, these are the files that the data, models, and plots will be saved in.

## Acknowledgements
A substantial portion of the code in pqc.py and quantum_model.py was taken from a tensorflow quantum tutorial at "https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning".
