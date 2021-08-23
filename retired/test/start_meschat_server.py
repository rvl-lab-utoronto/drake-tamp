#!/usr/bin/env python3
"""
This script is used to start a meshcat server, used for visualizing the 
TAMP experiments 
"""
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
import webbrowser
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
print(f"zmq url: {zmq_url}, the python command for experiments is:")
print(f"python run.py --url={zmq_url}")
print(f"web url: {web_url}, opening browser ...")
webbrowser.open(web_url, new = 2)

signal.signal(signal.SIGINT, signal_handler)
print('Press ctrl-c to stop the server.')
signal.pause()