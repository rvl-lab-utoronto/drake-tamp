"""
Simple module for capturing and handling statistics from a pddlstream solution
"""
import sys
import json
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

class CaptureOutput:
    """
    Captures interestting output from stdout and passes it into
    a file, it is also printed to stdout
    """

    def __init__(self, path):
        self.original = sys.stdout
        self.log = open(path, "a")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        pass


def process_pickle(picklepath, outpath):
    """ 
    Processes pickle output from pddlstream into a more
    human readable json format
    """
    pkl_file = open(picklepath , "rb")
    pkl_data = pickle.load(pkl_file)
    for subdict in pkl_data.values():
        if "distribution" in subdict:
            subdict.pop("distribution")
    json_file = open(outpath, "a")
    json.dump(pkl_data, json_file, indent=4, sort_keys=True)


class Iteration:
    """
    Class for recording information about an iteration
    """

    def __init__(self, datastr):
        self.data = {}
        t_list = datastr.split("|")
        for t in t_list:
            tsplit = t.split()
            num = None
            try:
                if "." in tsplit[-1]:
                    num = float(tsplit.pop(-1))
                else:
                    num = int(tsplit.pop(-1))
            except ValueError:
                pass
            name = " ".join(tsplit)
            self.data[name] = num

    def get_data(self):
        return self.data

    def get_total_time(self):
        return self.data["Total Time:"]

    def get_sample_time(self):
        return self.data["Sample Time:"]

    def get_evaluations(self):
        return self.data["Evaluations:"]

    def get_complexity(self):
        return self.data["Complexity:"]

class Attempt:
    """
    Class for recording information about an attempt
    """

    def __init__(self, iteration, datastr):
        self.iteration = iteration
        self.data = {}
        t_list = datastr.split("|")
        for t in t_list:
            tsplit = t.split()
            val = None
            try:
                end = tsplit.pop(-1)
                if "True" in end:
                    val = True
                elif "False" in end:
                    val = False
                elif "." in end:
                    val = float(end)
                else:
                    val = int(end)
            except ValueError:
                pass
            name = " ".join(tsplit)
            self.data[name] = val

    def get_data(self):
        return self.data

    def get_results(self):
        return self.data["Results:"]

    def get_time(self):
        return self.data["Time:"]

    def is_success(self):
        return self.data["Success:"]

class Summary:

    def __init__(self, datastr):
        self.data = {}
        datastr = datastr[datastr.index("{") + 1: datastr.index("}")]
        t_list = datastr.split(",")
        for t in t_list:
            tsplit = t.split()
            val = None
            try:
                end = tsplit.pop(-1)
                if "True" in end:
                    val = True
                elif "False" in end:
                    val = False
                elif "." in end:
                    val = float(end)
                else:
                    val = int(end)
            except ValueError:
                pass
            name = " ".join(tsplit)
            self.data[name] = val

    def get_complexity(self):
        return self.data["complexity:"]

    def get_evaluations(self):
        return self.data["evaluations:"]

    def get_total_time(self):
        return self.data["run_time:"]

    def get_sample_time(self):
        return self.data["sample_time:"]

def read_stats(filepath):
    last_iter = None
    iteration_map = OrderedDict()
    summary = None
    with open(filepath) as file:
        for line in file:
            if "Iteration" in line:
                last_iter = Iteration(datastr = line)
                iteration_map[last_iter] = []
            if "Attempt" in line:
                iteration_map[last_iter].append(Attempt(iteration = last_iter, datastr = line)) 
            if "Summary" in line:
                summary = Summary(datastr = line)


    return iteration_map, summary

            
def make_plot(filepath, save_path = None):
    """
    Makes a timing plot given the output to stdout of
    a call to pddlstream.algorithms.meta.solve
    stored in a file at `filepath`
    """
    # graph y axis: Results/Evaluations (#)
    # graph x axis: time
    # vertical lines denote each iteration
    # points for each (time,results), coming from each attempt
    # points for each (time,evaluations), coming from each iteration

    iteration_map, summary = read_stats(filepath)
    cum_time = 0


    fig, ax = plt.subplots(2,1)

    fail_results = [[], []]
    suc_results = [[], []]
    res = [[], []]

    iterations = 0
    end_attempt_time = 0

    complexity = [[],[]]
    evals = [[], []]

    prev_sample_time = next(iter(iteration_map)).get_sample_time()
    for iteration, at_list in iteration_map.items():
        time = iteration.get_total_time()
        sample_time = iteration.get_sample_time() - prev_sample_time
        prev_sample_time = iteration.get_sample_time()
    
        ax[0].axvspan(time - sample_time, time, color = "tab:blue", alpha = 0.2, label = "Sampling Time")
        ax[1].axvspan(time - sample_time, time, color = "tab:blue", alpha = 0.2)
        if len(evals[1]) > 0:
            evals[0].append(time - sample_time)
            evals[1].append(evals[1][-1])

        if iterations == (len(iteration_map) -1):
            ax[0].axvline(time, linestyle = ":", color = "tab:brown")
            ax[1].axvline(time, linestyle = ":", color = "tab:brown")
        else:
            ax[0].axvline(time, linestyle = ":", color = "tab:brown", label = "Iterations")
            ax[1].axvline(time, linestyle = ":", color = "tab:brown")
        evals[0].append(time)
        evals[1].append(iteration.get_evaluations())
        complexity[0].append(time)
        complexity[1].append(iteration.get_complexity())
        for attempt in at_list:
            time = iteration.get_total_time() +  attempt.get_time()
            res[0].append(time)
            res[1].append(attempt.get_results())
            if attempt.is_success():
                ax[0].axvline(time, linestyle = "--", color = "k", label = "Found Refined Optimistic Plan")
                suc_results[0].append(time)
                suc_results[1].append(attempt.get_results())
            else:
                fail_results[0].append(time)
                fail_results[1].append(attempt.get_results())
        #end_attempt_time = at_list[-1].get_time() + iteration.get_total_time()
        iterations += 1


    sample_time = summary.get_sample_time() - prev_sample_time
    ax[0].axvspan(summary.get_total_time() - sample_time, summary.get_total_time(), color = "tab:blue", alpha = 0.2)
    ax[1].axvspan(summary.get_total_time() - sample_time, summary.get_total_time(), color = "tab:blue", alpha = 0.2)

    evals[0].append(summary.get_total_time() - sample_time)
    evals[1].append(evals[1][-1])

    complexity[0].append(summary.get_total_time())
    complexity[1].append(complexity[1][-1])

    evals[0].append(summary.get_total_time())
    evals[1].append(summary.get_evaluations())
    ax[0].axvline(summary.get_total_time(), color = "tab:purple", label = "Found Solution")
    ax[1].axvline(summary.get_total_time(), color = "tab:purple", label = "Found Solution")
    res[0].append(summary.get_total_time())
    res[1].append(res[1][-1])

    ax[0].plot(suc_results[0], suc_results[1], linestyle = "", marker = "o", color = "tab:orange")
    ax[0].plot(fail_results[0], fail_results[1], linestyle = "", marker = "x", color = "tab:orange")
    ax[0].step(res[0], res[1], linestyle = "--", where = "post", marker = "", color = "tab:orange")
    ax[1].step(complexity[0], complexity[1],where = "pre", linestyle = "--", label = "complexity")

    fontsize = 10

    ax[0].set_xlabel("Time (s)", fontsize = fontsize)
    ax[0].set_ylabel("Number of Lifted Facts (Results)", fontsize = fontsize)
    ax[1].set_xlabel("Time (s)", fontsize = fontsize)
    ax[1].set_ylabel("Complexity", fontsize =fontsize)
    ax3 = ax[1].twinx()
    ax3.set_ylabel("Evaluations", fontsize = fontsize)
    ax3.plot(evals[0], evals[1], marker = "x", linestyle = "--", color = "tab:green", label = "Evalulations")
    ax[0].legend(fontsize = fontsize, loc = "lower right")
    ax[1].legend(loc = "upper left", fontsize = fontsize)
    ax3.legend(loc = "lower right", fontsize = fontsize)
    #plt.show()

    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    if save_path is not None:
        plt.savefig(save_path, dpi = 300)


