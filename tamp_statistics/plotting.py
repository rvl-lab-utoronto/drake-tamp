"""
Simple module for capturing and handling statistics from a pddlstream solution
"""
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


RESOLUTION = 0.1


def fix_time(times, start_time):
    return np.array(times) - np.ones(len(times)) * start_time


def make_plot(filepath, save_path=None):
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

    data = None
    with open(filepath) as json_stream:
        data = json.load(json_stream)

    fig, ax = plt.subplots(2, 1)
    start_time = data["start_time"]

    for i in range(2):
        for iteration in data["iterations"]:
            ax[i].axvline(
                iteration["total_time"], linestyle=":", color="k", linewidth=0.9
            )
        for attempt in data["attempts"]:
            if attempt["success"]:
                ax[i].axvline(
                    attempt["time"],
                    linestyle="-",
                    color="tab:purple",
                    label="Optimisitic Plan",
                )
    ax[0].axvline(data["summary"]["run_time"], color = "tab:pink", linewidth = 3, label = "Done")
    ax[1].axvline(data["summary"]["run_time"], color = "tab:pink", linewidth = 3)

    data["results"][0].append(data["summary"]["run_time"])
    data["results"][1].append(data["results"][1][-1])
    ax[0].step(
        data["results"][0],
        data["results"][1],
        where="post",
        linestyle="--",
        color="tab:blue",
        label="Results",
    )

    ax[1].step(
        data["evaluations"][0],
        np.array(data["evaluations"][1]),
        color="tab:orange",
        linestyle="-.",
        label="Evalulations",
        where="post",
    )

    ax2 = ax[1].twinx()

    ax2.step(
        data["complexity"][0],
        np.array(data["complexity"][1]),
        color="tab:blue",
        linestyle="--",
        where="post",
        label="Complexity",
    )

    curr_interval = data["sampling_intervals"][0]
    for next_interval in data["sampling_intervals"][1:]:
        if np.isclose(next_interval[0], curr_interval[1], rtol=0, atol=RESOLUTION):
            curr_interval[1] = next_interval[1]
        else:
            ax[0].axvspan(
                curr_interval[0],
                curr_interval[1],
                color="tab:green",
                alpha=0.2,
                label="Sampling Time",
            )
            ax[1].axvspan(
                curr_interval[0], curr_interval[1], color="tab:green", alpha=0.2
            )
            curr_interval = next_interval

    ax[0].axvspan(
        curr_interval[0],
        curr_interval[1],
        color="tab:green",
        alpha=0.2,
        label="Sampling Time",
    )
    ax[1].axvspan(
        curr_interval[0], curr_interval[1], color="tab:green", alpha=0.2
    )

    ax[0].set_xlabel("Time (s)")
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Results (Lifted Facts)")
    ax[1].set_ylabel("Evaluations (Grounded Facts)")
    ax2.set_ylabel("Complexity")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="upper left")
    ax2.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
