import importlib, pkg_resources
importlib.reload(pkg_resources)

from helper import *
from fox_in_a_hole import *

import cirq
import time
import numpy as np
from collections import deque
tf.get_logger().setLevel('ERROR')

class QDQN():
    def __init__(self, savename, model, model_target, n_layers, n_holes, qubits, memory_size, learning_rates, gamma, n_episodes, steps_per_train, soft_weight_update, steps_per_target_update, tau, epsilon_start, epsilon_min, decay_epsilon, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy):
        self.savename = savename
        self.model = model
        self.model_target = model_target
        self.n_layers = n_layers
        self.n_holes = n_holes
        self.qubits = qubits
        self.memory_size = memory_size
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.steps_per_train = steps_per_train
        self.soft_weight_update = soft_weight_update
        self.steps_per_target_update = steps_per_target_update
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_epsilon = decay_epsilon
        self.temperature = temperature
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.max_size_buffer = max_size_buffer
        self.exploration_strategy = exploration_strategy
        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=learning_rates[0], amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rates[1], amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rates[2], amsgrad=True)
        # Indexes of the weights for each of the parts of the circuit
        self.w_in, self.w_var, self.w_out = 1, 0, 2

    def main(self):
        env = FoxInAHole(self.n_holes, self.memory_size)
        replay_memory = deque(maxlen=self.max_size_buffer)

        episode_reward_history = []
        episode_length_history = []
        step_count = 0

        for episode in range(self.n_episodes):
            episode_reward = 0
            state = deque([0] * self.memory_size, maxlen=self.memory_size)
            done = env.reset()
            episode_length = 0

            # annealing, done before the while loop because the first episode equals 0 so it returns the original epsilon back
            if self.exploration_strategy == 'egreedy':
                epsilon = exponential_anneal(episode, self.epsilon_start, self.epsilon_min, self.decay_epsilon)

            while True:
                episode_length += 1

                state_tensor = tf.convert_to_tensor([np.array(state)])
                q_vals = self.model([state_tensor])
                possible_actions = np.arange(self.n_holes)

                # Sample action
                if self.exploration_strategy == 'egreedy':
                    coin = np.random.random()
                    if coin > epsilon:
                        action = int(tf.argmax(q_vals[0]).numpy())
                    else:
                        action = np.random.randint(self.n_holes)
                elif self.exploration_strategy == 'boltzmann':
                    probabilities = boltzmann_exploration(np.array(q_vals), self.temperature)
                    action = np.random.choice(possible_actions, p=probabilities)

                # Interact with env
                interaction = self.interact_env(state, action, env)

                # Store interaction in the replay memory
                replay_memory.append(interaction)

                state = interaction['next_state']
                episode_reward += interaction['reward']
                step_count += 1

                # Update model
                if step_count % self.steps_per_train == 0 and len(replay_memory) >= self.min_size_buffer:
                    # Sample a batch of interactions and update Q_function
                    training_batch = np.random.choice(replay_memory, size=self.batch_size)
                    self.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                      np.asarray([x['action'] for x in training_batch]),
                                      np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                      np.asarray([x['next_state'] for x in training_batch]),
                                      np.asarray([x['done'] for x in training_batch], dtype=np.float32))

                # Update target model
                if self.soft_weight_update:
                    self.update_model()
                elif step_count % self.steps_per_target_update == 0:
                    self.update_model()

                # Check if the episode is finished
                if interaction['done']:
                    break

                env.step()

            episode_length_history.append(episode_length)
            episode_reward_history.append(episode_reward)

            if episode % 50 == 0:
                print('Training progress: ' + str(episode) + '/' + str(self.n_episodes))

        if self.savename != False:
            self.save_data(episode_reward_history, episode_length_history)

    def update_model(self):
        '''
        Copies weights from base model to target network.
        param base_model:       tf base model
        param target_network:   tf target network
        '''
        if self.soft_weight_update:
            new_weights = []
            for TN_layer, BM_layer in zip(self.model_target.get_weights(), self.model.get_weights()):
                new_weights.append((1-self.tau) * TN_layer + self.tau * BM_layer)
            self.model_target.set_weights(new_weights)
        else:
            self.model_target.set_weights(self.model.get_weights())

    def save_data(self, rewards, episode_lengths):
        data = {'n_holes': self.n_holes, 'rewards': rewards, 'episode_lengths': episode_lengths, 'n_layers': self.n_layers}
        np.save('data/' + self.savename + '.npy', data)
        self.model_target.save_weights('models/' + self.savename)

    def interact_env(self, state, action, env):
        # Apply sampled action in the environment, receive reward and next state
        reward, done = env.guess(action)
        next_state = state.copy()
        next_state.append(action)

        interaction = {'state': state, 'action': action, 'next_state': next_state,
                       'reward': reward, 'done':np.float32(done)}

        return interaction

    @tf.function
    def Q_learning_update(self, states, actions, rewards, next_states, done):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = self.model_target([next_states])
        target_q_values = rewards + (self.gamma * tf.reduce_max(future_rewards, axis=1)
                                                       * (1.0 - done))
        masks = tf.one_hot(actions, self.n_holes)

        # Train the model on the states and target Q-values
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.MeanSquaredError()(target_q_values, q_values_masked)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

def main():
    n_holes = 5
    memory_size = 2 * (n_holes - 2)
    n_qubits = memory_size
    n_layers = 5  # Number of layers in the PQC

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = []
    for i in range(n_holes):
        observables.append(ops[i])

    n_episodes = 100
    learning_rates = [0.001, 0.001, 0.1]
    gamma = 1

    # Define replay memory
    max_size_buffer = 10000 # Maximum replay length
    min_size_buffer = 1000

    epsilon_start = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.01  # Minimum epsilon greedy parameter
    decay_epsilon = 0.01 # Decay rate of epsilon greedy parameter
    temperature = 0.01 # Temperature parameter for the Boltzmann exploration
    batch_size = 64
    steps_per_train = 10 # Train the model every x steps
    steps_per_target_update = 20 # parameter for hard updating target network weights
    tau = 0.05 # parameter for soft updating target network weights
    soft_weight_update = False # boolean that decides whether to soft update or hard update the target network weights
    exploration_strategy = 'boltzmann' # 'egreedy' or 'boltzmann'

    savename = 'test'

    start = time.time()

    quantum_model = QuantumModel(qubits, n_layers, observables)

    model = quantum_model.generate_model_Qlearning(False)
    model_target = quantum_model.generate_model_Qlearning(True)

    model_target.set_weights(model.get_weights())

    qdqn = QDQN(savename, model, model_target, n_layers, n_holes, qubits, memory_size, learning_rates, gamma, n_episodes, steps_per_train, soft_weight_update, steps_per_target_update, tau, epsilon_start, epsilon_min, decay_epsilon, temperature, batch_size, min_size_buffer, max_size_buffer, exploration_strategy)

    qdqn.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))

if __name__ == '__main__':
    main()