import importlib, pkg_resources
importlib.reload(pkg_resources)

from helper import *
import tensorflow_quantum as tfq

import cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))


n_holes = 5
memory_size = 2*(n_holes-2)
n_qubits = memory_size
n_layers = 5 # Number of layers in the PQC

qubits = cirq.GridQubit.rect(1, n_qubits)
ops = [cirq.Z(q) for q in qubits]
observables = []
for i in range(n_holes):
    observables.append(ops[i])

def generate_model_Qlearning(qubits, n_layers, observables, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model

def update_model(base_model, target_model, soft_update, tau):
    '''
    Copies weights from base model to target network.
    param base_model:       tf base model
    param target_network:   tf target network
    '''
    if soft_update:
        new_weights = []
        for TN_layer, BM_layer in zip(target_model.get_weights(), base_model.get_weights()):
            new_weights.append((1-tau) * TN_layer + tau * BM_layer)
        target_model.set_weights(new_weights)
    else:
        target_model.set_weights(base_model.get_weights())

def save_data(model_target, savename, rewards, episode_lengths):
    data = {'n_holes': n_holes, 'rewards': rewards, 'episode_lengths': episode_lengths}
    np.save('data/' + savename + '.npy', data)
    model_target.save_weights('models/' + savename)

model = generate_model_Qlearning(qubits, n_layers, observables, False)
model_target = generate_model_Qlearning(qubits, n_layers, observables, True)

model_target.set_weights(model.get_weights())

def interact_env(state, action, env):
    # Apply sampled action in the environment, receive reward and next state
    reward, done = env.guess(action)
    next_state = state.copy()
    next_state.append(action)

    interaction = {'state': state, 'action': action, 'next_state': next_state,
                   'reward': reward, 'done':np.float32(done)}

    return interaction

@tf.function
def Q_learning_update(states, actions, rewards, next_states, done, model, gamma):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)
    done = tf.convert_to_tensor(done)

    # Compute their target q_values and the masks on sampled actions
    future_rewards = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(future_rewards, axis=1)
                                                   * (1.0 - done))
    masks = tf.one_hot(actions, n_holes)

    # Train the model on the states and target Q-values
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.MeanSquaredError()(target_q_values, q_values_masked)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


gamma = 1
n_episodes = 100

# Define replay memory
max_memory_length = 10000 # Maximum replay length
min_memory_length = 0
replay_memory = deque(maxlen=max_memory_length)

epsilon_start = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
decay_epsilon = 0.01 # Decay rate of epsilon greedy parameter
temperature = 0.01 # Temperature parameter for the Boltzmann exploration
batch_size = 64
steps_per_update = 10 # Train the model every x steps
steps_per_target_update = 20 # parameter for hard updating target network weights
tau = 0.05 # parameter for soft updating target network weights
soft_weight_update = False # boolean that decides whether to soft update or hard update the target network weights
exploration_strategy = 'boltzmann' # 'egreedy' or 'boltzmann'

savename = 'test'

optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2

env = FoxInAHole(n_holes, memory_size)

episode_reward_history = []
episode_length_history = []
step_count = 0

for episode in range(n_episodes):
    episode_reward = 0
    state = deque([0] * memory_size, maxlen=memory_size)
    done = env.reset()
    episode_length = 0

    # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
    if exploration_strategy == 'egreedy':
        epsilon = exponential_anneal(episode, epsilon_start, epsilon_min, decay_epsilon)

    while True:
        episode_length += 1

        state_tensor = tf.convert_to_tensor([np.array(state)])
        q_vals = model([state_tensor])
        possible_actions = np.arange(n_holes)

        # Sample action
        if exploration_strategy == 'egreedy':
            coin = np.random.random()
            if coin > epsilon:
                action = int(tf.argmax(q_vals[0]).numpy())
            else:
                action = np.random.randint(n_holes)
        elif exploration_strategy == 'boltzmann':
            probabilities = boltzmann_exploration(np.array(q_vals), temperature)
            action = np.random.choice(possible_actions, p=probabilities)

        # Interact with env
        interaction = interact_env(state, action, env)

        # Store interaction in the replay memory
        replay_memory.append(interaction)

        state = interaction['next_state']
        episode_reward += interaction['reward']
        step_count += 1

        # Update model
        if step_count % steps_per_update == 0 and len(replay_memory) >= min_memory_length:
            # Sample a batch of interactions and update Q_function
            training_batch = np.random.choice(replay_memory, size=batch_size)
            Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                              np.asarray([x['action'] for x in training_batch]),
                              np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                              np.asarray([x['next_state'] for x in training_batch]),
                              np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                              model, gamma)

        # Update target model
        if soft_weight_update:
            update_model(base_model=model, target_model=model_target, soft_update=soft_weight_update, tau=tau)
        elif step_count % steps_per_target_update == 0:
            update_model(base_model=model, target_model=model_target, soft_update=soft_weight_update, tau=tau)

        # Check if the episode is finished
        if interaction['done']:
            break

        env.step()

    episode_length_history.append(episode_length)
    episode_reward_history.append(episode_reward)

    if episode % 50 == 0:
        print('Training progress: '+str(episode)+'/'+str(n_episodes))

if savename != False:
    save_data(model_target, savename, episode_reward_history, episode_length_history)