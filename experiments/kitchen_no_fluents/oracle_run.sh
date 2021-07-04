#!/bin/bash

oracle="False"
default_unique="False"
mode="normal"
url=""
problem=""

usage() {
    echo "Usage: $0 [-s] [-h]"
    echo "Use -s if you want to save the stats.json to drake-tamp/learning/data, running without the oracle"
    echo "Use -o if you want to run the oracle"
    echo "Use -u <url> to provide a meshcat url"
    exit 1 
}

while getopts sou: option
do
    case "${option}"
        in
        s) 
            oracle="False"
            default_unique="False"
            echo "Saving stats.json to drake-tamp/learning/data"
            mode="save"
            ;;
        o)
            oracle="True"
            default_unique="True"
            echo "Using oracle"
            mode="oracle"
            ;;
        u)
            url=${OPTARG};;
        *)
            usage
            ;;
    esac
done

echo "ORACLE=$oracle"
echo "DEFAULT_UNIQUE=$default_unique"

#export ORACLE=$oracle
#export DEFAULT_UNIQUE=$default_unique

if [[ ! $url ]]; then
    python run.py --mode=$mode
else
    python run.py --mode=$mode --url=$url
fi