from quantum_model import *

# amount of repetitions that will be averaged over for the experiment
repetitions = 20
# amount of episodes that will run
n_episodes = 5000
# game parameters
n_holes = 5
memory_size = 2*(n_holes-2)
n_qubits = memory_size
n_layers = 7  # Number of layers in the PQC

qubits = cirq.GridQubit.rect(1, n_qubits)
ops = [cirq.Z(q) for q in qubits]
observables = []
for i in range(n_holes):
    observables.append(ops[i])

# Hyperparameters of the algorithm and other parameters of the program
learning_rate_in = 0.0001
learning_rate_var = 0.01
learning_rate_out = 0.1
gamma = 1  # discount factor
epsilon_start = 1  # 100%
epsilon_min = 0.01  # 1%
steps_per_train = 10 # Per how many game steps a singular batch of model training happens
soft_weight_update = True # False if hard updating
steps_per_target_update = 10 # This parameter only matters when the weights are being hard updated
tau = 0.05
decay_epsilon = 0.01  # the amount with which the exploration parameter changes after each episode
temperature = 0.01
batch_size = 64
min_size_buffer = 1000
max_size_buffer = 10000
exploration_strategy = 'egreedy' # 'egreedy' or 'boltzmann'

data_names = []

start = time.time()

savename = 'holes_'+str(n_holes)
for rep in range(repetitions):
    quantum_model = QuantumModel(qubits, n_layers, observables)

    model = quantum_model.generate_model_Qlearning(False)
    model_target = quantum_model.generate_model_Qlearning(True)

    model_target.set_weights(model.get_weights())

    file_name = savename + '-repetition_' + str(rep + 1)

    qdqn = QDQN(file_name, model, model_target, n_layers, n_holes, qubits, memory_size, [learning_rate_in, learning_rate_var, learning_rate_out], gamma, n_episodes,
                steps_per_train, soft_weight_update, steps_per_target_update, tau, epsilon_start, epsilon_min,
                decay_epsilon, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy)

    qdqn.main()

    data_names.append(file_name)

    print('Finished repetition '+str(rep+1)+'/'+str(repetitions))

plot_averaged(data_names=data_names, show=False, savename=savename, smooth=False)
plot_averaged(data_names=data_names, show=False, savename=savename+'-smooth', smooth=True)

data_names = []

end = time.time()

print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))