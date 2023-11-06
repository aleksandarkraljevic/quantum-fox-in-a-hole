import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
from fox_in_a_hole import *

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
    if smooth==True:
        rewards = savgol_filter(rewards, 31, 1)
    episodes = np.arange(1, len(rewards) + 1)
    dataframe = np.vstack((rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])
    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='episodes', y='reward')
    plt.title('Reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename)
    if show:
        plt.show()

def plot_averaged(data_names, show, savename, smooth):
    n_names = len(data_names)
    data = np.load('data/'+data_names[0]+'.npy', allow_pickle=True)
    memory_size = data.item().get('memory_size')
    rewards = data.item().get('rewards')
    episodes = np.arange(1, len(rewards) + 1)
    for i in range(n_names-1):
        data =  np.load('data/'+data_names[i+1]+'.npy', allow_pickle=True)
        new_rewards = data.item().get('rewards')
        rewards = np.vstack((rewards, new_rewards))
    mean_rewards = np.mean(rewards, axis=0)
    se_rewards = np.std(rewards, axis=0) / np.sqrt(n_names) # standard error
    lower_bound = np.clip(mean_rewards-se_rewards, -1*memory_size , 1)
    upper_bound = np.clip(mean_rewards+se_rewards, -1*memory_size, 1)
    if smooth == True:
        mean_rewards = savgol_filter(mean_rewards, 51, 1)
    dataframe = np.vstack((mean_rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])

    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='episodes', y='reward')
    plt.fill_between(episodes, lower_bound, upper_bound, color='b', alpha=0.2)
    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename+'.png')
    if show:
        plt.show()

def compare_models(parameter_names, repetitions, show, savename, label_names, smooth):
    # this function requires the user to put all the experiment data in the data folder
    plt.figure()
    sns.set_theme()

    for experiment in range(len(parameter_names)):
        data = np.load('data/'+parameter_names[experiment]+'-repetition_1.npy', allow_pickle=True)
        memory_size = data.item().get('memory_size')
        rewards = data.item().get('rewards')
        episodes = np.arange(1, len(rewards) + 1)
        for i in range(repetitions-1):
            data = np.load('data/'+parameter_names[experiment]+'-repetition_'+str(i+2)+'.npy', allow_pickle=True)
            new_rewards = data.item().get('rewards')
            rewards = np.vstack((rewards, new_rewards))
        mean_rewards = np.mean(rewards, axis=0)
        se_rewards = np.std(rewards, axis=0) / np.sqrt(repetitions)  # standard error
        lower_bound = np.clip(mean_rewards - se_rewards, -1 * memory_size, 1)
        upper_bound = np.clip(mean_rewards + se_rewards, -1 * memory_size, 1)
        if smooth == True:
            mean_rewards = savgol_filter(mean_rewards, 51, 1)
        dataframe = np.vstack((mean_rewards, episodes)).transpose()
        dataframe = pd.DataFrame(data=dataframe, columns=['reward', 'episodes'])

        sns.lineplot(data=dataframe, x='episodes', y='reward', label=label_names[experiment])
        plt.fill_between(episodes, lower_bound, upper_bound, alpha=0.3)

    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/' + savename + '.png')
    if show:
        plt.show()

def evaluate(model_name, n_samples, print_strategy, print_evaluation):
    model = tf.keras.models.load_model('models/'+model_name+'.keras')
    data = np.load('data/'+model_name+'.npy', allow_pickle=True)
    n_holes = data.item().get('n_holes')
    memory_size = data.item().get('memory_size')
    env = FoxInAHole(n_holes, memory_size)
    observation = [0] * memory_size
    done = env.reset()
    won, lost = done
    current_episode_length = 0
    episode_lengths = []
    rewards = []
    if print_strategy:
        for step in range(len(observation)):
            predicted_q_values = model(np.asarray(observation).reshape(1, memory_size))
            action = np.argmax(predicted_q_values) + 1
            observation[step] = action
        print(observation)
        observation = [0] * memory_size
    for sample in range(n_samples):
        while (not won) and (not lost):
            current_episode_length += 1
            predicted_q_values = model(np.asarray(observation).reshape(1, memory_size))
            action = np.argmax(predicted_q_values) + 1
            reward, done = env.guess(action, current_episode_length)
            won, lost = done
            new_observation = observation.copy()
            new_observation[current_episode_length - 1] = action
            observation = new_observation
            env.step()
        episode_lengths.append(current_episode_length)
        rewards.append(reward)
        current_episode_length = 0
        done = env.reset()
        won, lost = done
        observation = [0] * memory_size

    if print_evaluation:
        print('The average amount of guesses needed to finish the game is: ',round(np.mean(episode_lengths),2))
        print('The average reward per game after '+str(n_samples)+' games is: ',round(np.mean(rewards),2))

    return np.mean(episode_lengths)