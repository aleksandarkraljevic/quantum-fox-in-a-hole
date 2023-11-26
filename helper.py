import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque
from scipy.signal import savgol_filter
from fox_in_a_hole import *
from pqc import *

def exponential_anneal(t, start, final, decay_constant):
    '''
    Exponential annealing scheduler for epsilon-greedy policy.
    param t:        current timestep
    param start:    initial value
    param final:    value after percentage*T steps
    '''
    return final + (start - final) * np.exp(-decay_constant*t)


def boltzmann_exploration(actions, temperature):
    '''
    Boltzmann exploration policy.
    param actions:      vector with possible actions
    param temperature:  exploration parameter
    return:             vector with probabilities for choosing each option
    '''
    actions = actions[0] - np.max(actions[0])
    a = actions / temperature  # scale by temperature
    return np.exp(a)/np.sum(np.exp(a))

def plot(data_name, show, savename, smooth):
    data = np.load('data/'+data_name+'.npy', allow_pickle=True)
    rewards = data.item().get('rewards')
    n_holes = data.item().get('n_holes')
    memory_size = 2 * (n_holes - 2)
    if smooth==True:
        rewards = savgol_filter(rewards, 71, 1)
    episodes = np.arange(1, len(rewards) + 1)
    dataframe = np.vstack((rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])
    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='episodes', y='reward')
    plt.ylim(-1 * memory_size, 0)
    plt.title('Reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename+'.png')
    if show:
        plt.show()

def plot_averaged(data_names, show, savename, smooth):
    n_names = len(data_names)
    data = np.load('data/'+data_names[0]+'.npy', allow_pickle=True)
    n_holes = data.item().get('n_holes')
    memory_size = 2 * (n_holes - 2)
    rewards = data.item().get('rewards')
    episodes = np.arange(1, len(rewards) + 1)
    for i in range(n_names-1):
        data =  np.load('data/'+data_names[i+1]+'.npy', allow_pickle=True)
        new_rewards = data.item().get('rewards')
        rewards = np.vstack((rewards, new_rewards))
    mean_rewards = np.mean(rewards, axis=0)
    se_rewards = np.std(rewards, axis=0) / np.sqrt(n_names) # standard error
    lower_bound = np.clip(mean_rewards-se_rewards, None , 0)
    upper_bound = np.clip(mean_rewards+se_rewards, None, 0)
    if smooth == True:
        mean_rewards = savgol_filter(mean_rewards, 71, 1)
    dataframe = np.vstack((mean_rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])

    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='episodes', y='reward')
    plt.fill_between(episodes, lower_bound, upper_bound, color='b', alpha=0.2)
    plt.ylim(-1*memory_size,0)
    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename+'.png')
    if show:
        plt.show()

def compare_models(parameter_names, repetitions, show, savename, label_names, smooth):
    # this function requires the user to put all the experiment data in the data folder
    colors_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    plt.figure()
    sns.set_theme()

    for experiment in range(len(parameter_names)):
        data = np.load('data/'+parameter_names[experiment]+'-repetition_1.npy', allow_pickle=True)
        n_holes = data.item().get('n_holes')
        memory_size = 2*(n_holes-2)
        rewards = data.item().get('rewards')
        episodes = np.arange(1, len(rewards) + 1)
        for i in range(repetitions-1):
            data = np.load('data/'+parameter_names[experiment]+'-repetition_'+str(i+2)+'.npy', allow_pickle=True)
            new_rewards = data.item().get('rewards')
            rewards = np.vstack((rewards, new_rewards))
        mean_rewards = np.mean(rewards, axis=0)
        se_rewards = np.std(rewards, axis=0) / np.sqrt(repetitions)  # standard error
        lower_bound = np.clip(mean_rewards - se_rewards, None, 0)
        upper_bound = np.clip(mean_rewards + se_rewards, None, 0)
        if smooth == True:
            mean_rewards = savgol_filter(mean_rewards, 71, 1)
        dataframe = np.vstack((mean_rewards, episodes)).transpose()
        dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])

        sns.lineplot(data=dataframe, x='episodes', y='reward', label=label_names[experiment])
        plt.fill_between(episodes, lower_bound, upper_bound, color=colors_list[experiment], alpha=0.1)
        plt.ylim(-1 * memory_size, 0)

    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/' + savename + '.png')
    if show:
        plt.show()

def evaluate(model_name, n_samples, print_strategy, print_evaluation):
    data = np.load('data/'+model_name+'.npy', allow_pickle=True)
    n_holes = data.item().get('n_holes')
    n_layers = data.item().get('n_layers')
    memory_size = 2 * (n_holes - 2)
    qubits = cirq.GridQubit.rect(1, memory_size)
    ops = [cirq.Z(q) for q in qubits]
    observables = []
    for i in range(n_holes):
        observables.append(ops[i])
    quantum_model = QuantumModel(qubits, n_layers, observables)
    model = quantum_model.generate_model_Qlearning(False)
    model.load_weights('models/' + model_name)
    env = FoxInAHole(n_holes, memory_size)
    episode_lengths = []
    episode_rewards = []
    if print_strategy:
        done = env.reset()
        state = deque([0]*memory_size, maxlen=memory_size)
        for step in range(memory_size):
            state_tensor = tf.convert_to_tensor([np.array(state)])
            q_vals = model([state_tensor])
            action = np.argmax(q_vals) + 1
            state.append(action)
        print(state)
    for sample in range(n_samples):
        current_episode_length = 0
        episode_reward = 0
        done = env.reset()
        state = deque([0]*memory_size, maxlen=memory_size)
        while not done:
            current_episode_length += 1
            state_tensor = tf.convert_to_tensor([np.array(state)])
            q_vals = model([state_tensor])
            action = np.argmax(q_vals)
            reward, done = env.guess(action)
            episode_reward += reward
            new_observation = state.copy()
            new_observation.append(action)
            state = new_observation
            env.step()
        episode_lengths.append(current_episode_length)
        episode_rewards.append(episode_reward)

    if print_evaluation:
        print('The average amount of guesses needed to finish the game is: ',round(np.mean(episode_lengths),2))
        print('The average reward per game after '+str(n_samples)+' games is: ',round(np.mean(episode_rewards),2))

    return np.mean(episode_lengths)