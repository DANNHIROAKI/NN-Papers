
import os
import numpy as np
from contextlib import redirect_stdout


def print_results(accuracies, setup_times, training_times, *dictionaries, 
                      architecture=None, dataset=None, print_all=False, **kwargs):
    r"""Print a few lines summarizing the results of a named architecture on a
    named dataset. If additionally dictionaries and/or keyword arguments are
    given, all their key-value pairs are also printed. If print_all is True,
    the individual results of all runs are also printed.
    """
    
    def p(varname, var, unit=''):
        if var is not None:
            if np.size(var) > 1:
                print(" - Average {}: {} {} +- {}".format(varname, np.mean(var), unit, np.std(var)))
            else:
                print(" - {}: {} {}".format(varname, np.mean(var), unit))
                
    N = len(accuracies)
    if N == 0:
        return
        
    if architecture is not None:
        print("Summary of {} runs with architecture \"{}\"".format(N, architecture)
              + (":" if dataset is None else " on dataset \"{}\":".format(dataset)))
    
    p("Accuracy", 100*np.array(accuracies), '%')
    p("Setup Time", setup_times, 's')
    p("Training Time", training_times, 's')
    if setup_times is not None and training_times is not None:
        p("Total Time", np.add(setup_times, training_times), 's')
    
    if len(dictionaries) > 0 or len(kwargs) > 0:
        print()
        dictionaries = list(dictionaries)
        dictionaries.append(kwargs)
        for d in dictionaries:
            if not isinstance(d, dict):
                d = d.__dict__
            for key, val in d.items():
                print(' - {} = {}'.format(key, val))
    
    if N > 1 and print_all:
        print()
        print("Individual run results:")
        if isinstance(setup_times, list):
            print(" Run   SetupTime  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {: 12.4f}  {:.4f}".format(run+1, setup_times[run], training_times[run], 100*accuracies[run]))
        else:
            print(" Run  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {:.4f}".format(run+1, training_times[run], 100*accuracies[run]))
            

def save_results(dir, dataset, architecture, *args, **kwargs):
    r"""Save the output of print_results in a file. The .TXT file is named 
    after the architecture and created in a directory named after the dataset 
    within the given parent directory."""
    dir = os.path.join(dir, dataset)
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, architecture + '.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            print_results(*args, architecture=architecture, dataset=dataset, print_all=True, **kwargs)
    print('Results saved to file {}'.format(filename))