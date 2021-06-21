"""
Simple module for capturing and handling statistics from a pddlstream solution
"""
import sys
import json
import pickle

class CaptureOutput:
    """
    Captures interestting output from stdout and passes it into
    a file, it is also printed to stdout
    """

    def __init__(self, path, filt = None):
        self.original = sys.stdout
        self.log = open(path, "a")
        self.stdout = sys.stdout
        self.filt = filt

    def write(self, message):
        self.stdout.write(message)
        if self.filt is not None:
            for f in self.filt:
                if f in message:
                    continue
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
