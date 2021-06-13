"""
This module contains simple utility functions 
"""

class Colors:
    """
    This class is used to print text to the terminal in color. 

    Basic Usage:
    print(f"{Colors.COLOR}my text{Colors.RESET}")
    """
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"