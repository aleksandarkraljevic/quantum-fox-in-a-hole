from helper import *

def evaluate_experiment(parameter_name, repetitions, n_samples, print_strategies, print_evaluation):
    '''
    Evaluates an experiment by looking at each of its repetition's found strategy, and averaging its performance.

    Parameters
    ----------
    parameter_name (str):
        The name of the experiment files, excluding "-repeition_" onwards.
    repetitions (int):
        The number of repetitions that the experiment contains.
    n_samples (int):
        The amount of samples that the model of each repetition will be averaged over.
    print_strategies (boolean):
        Whether the strategy of each model will be printed.
    print_evaluation (boolean):
        Whether the performance of each model will be printed.
    '''
    mean_lengths = []

    for rep in range(repetitions):
        name = parameter_name + '-repetition_' + str(rep + 1)
        print('For repetition ' + str(rep + 1) + ':')
        mean_length = evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategies, print_evaluation=print_evaluation, plot_distribution=False, save=False)
        mean_lengths.append(mean_length)

    print('Average amount of guesses needed over all repetitions is:', round(np.mean(mean_lengths),2), '+- ', round(np.std(mean_lengths)/np.sqrt(repetitions),2))

def plot_experiment(parameter_name, repetitions, show, save):
    '''
    Plots an experiment. Both raw plots and smoothed plots are performed.

    Parameters
    ----------
    parameter_name (str):
        The name of the experiment files, excluding "-repeition_" onwards.
    repetitions (int):
        The number of repetitions that the experiment contains.
    show (boolean):
        Whether the plots will be shown to the user.
    save (boolean):
        Whether the plots will be saved using the same name as parameter_name.
    '''
    data_names = []

    for rep in range(repetitions):
        data_names.append(parameter_name + '-repetition_' + str(rep + 1))

    if save:
        plot_averaged(data_names=data_names, show=show, savename=parameter_name, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=parameter_name+'-smooth', smooth=True)
    else:
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=False)
        plot_averaged(data_names=data_names, show=show, savename=False, smooth=True)

def evaluate_model(name, n_samples, print_strategy, print_evaluation, plot_model, show, save, plot_distribution):
    '''
    Evaluates a single model by looking at its best-found strategy, and averaging its performance. It also plots the distribution of all the numbers of needed guesses after averaging.

    Parameters
    ----------
    name (str):
        The name of the model file, excluding the file extension.
    n_samples (int):
        The amount of samples that the model will be averaged over.
    print_strategy (boolean):
        Whether the strategy of the model will be printed.
    print_evaluation (boolean):
        Whether the performance of the model will be printed.
    plot_model (boolean):
        Whether the model's training will be plotted.
    show (boolean):
        Whether the plots will be shown to the user.
    save (boolean):
        Whether the plots will be saved using the same name as parameter_name.
    plot_distribution (boolean):
        Whether the distribution of the numbers of needed guesses of all samples will be plotted.
    '''
    evaluate(model_name=name, n_samples=n_samples, print_strategy=print_strategy, print_evaluation=print_evaluation, plot_distribution=plot_distribution, save=save)
    if plot_model:
        if save:
            plot(data_name=name, show=show, savename=name, smooth=False)
            plot(data_name=name, show=show, savename=name+'-smooth', smooth=True)
        else:
            plot(data_name=name, show=show, savename=False, smooth=False)
            plot(data_name=name, show=show, savename=False, smooth=True)

def main():
    '''
    This function evalutes what the user is interested in evaluating. Each of the following lines can be commented or uncommented depending on what the user exactly wants to evaluate
    '''
    parameter_names = ['layers_9', 'layers_10']
    label_names = ['layers_9', 'layers_10']
    parameter_name = 'holes_8'

    evaluate_model(parameter_name+'-repetition_16', 10000, False, False, False, False, True, True)

    #compare_models(parameter_names=parameter_names, repetitions=20, show=True, savename='compare_layers_9_10', label_names=label_names, smooth=True)

    #evaluate_experiment(parameter_name=parameter_name, repetitions=20, n_samples=10000, print_strategies=True, print_evaluation=True)

    #plot_experiment(parameter_name, 20, True, True)


if __name__ == '__main__':
    main()