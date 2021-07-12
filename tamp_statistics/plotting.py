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


def make_plot(filepath, save_path=None, show=False):
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
                iteration["total_time"],
                linestyle=":",
                color="tab:brown",
                label="Iteration",
            )
        for attempt in data["attempts"]:
            if attempt["success"]:
                ax[i].axvline(
                    attempt["time"],
                    color="k",
                    label="Optimisitic Plan",
                )

    for time in data["unrefined"]:
        ax[0].axvline(time, color="k", linestyle = "--", label="Unrefined Plan")
        ax[1].axvline(
            time,
            color="k",
            linestyle = "--"
        )

    ax[0].axvline(
        data["summary"]["run_time"], color="tab:blue", linewidth=3, label="Done"
    )
    ax[1].axvline(data["summary"]["run_time"], color="tab:blue", linewidth=3)

    data["results"][0].append(data["summary"]["run_time"])
    data["results"][1].append(data["results"][1][-1])
    ax[0].step(
        data["results"][0],
        data["results"][1],
        where="post",
        linestyle="--",
        color="tab:orange",
        label="Results",
    )

    ax[1].step(
        data["evaluations"][0],
        np.array(data["evaluations"][1]),
        color="tab:green",
        linestyle="--",
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

    if len(data["sampling_intervals"]) == 0:
        print("No real plan, cannot plot")
        return 
    curr_interval = data["sampling_intervals"][0]
    for next_interval in data["sampling_intervals"][1:]:
        if np.isclose(next_interval[0], curr_interval[1], rtol=0, atol=RESOLUTION):
            curr_interval[1] = next_interval[1]
        else:
            ax[0].axvspan(
                curr_interval[0],
                curr_interval[1],
                color="tab:blue",
                alpha=0.2,
                label="Sampling Time",
            )
            ax[1].axvspan(
                curr_interval[0], curr_interval[1], color="tab:blue", alpha=0.2
            )
            curr_interval = next_interval

    ax[0].axvspan(
        curr_interval[0],
        curr_interval[1],
        color="tab:blue",
        alpha=0.2,
        label="Sampling Time",
    )
    ax[1].axvspan(curr_interval[0], curr_interval[1], color="tab:blue", alpha=0.2)

    ax[0].set_xlabel("Time (s)")
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Results (Lifted Facts)")
    ax[1].set_ylabel("Evaluations (Grounded Facts)")
    ax2.set_ylabel("Complexity")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="upper left")
    ax2.legend(loc="lower right")

    if save_path is not None:
        fig.tight_layout()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

    plt.close(fig)
