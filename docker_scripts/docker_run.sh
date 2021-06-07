#!/bin/bash

echo ""
echo "Running drake_workspace container"
echo ""

port=2300
name='drake_workspace'

while getopts p:n: option
do
    case "${option}"
        in
        p) port=${OPTARG};; 
        n) name=${OPTARG};;
    esac
done

echo "Using port: $port"
echo "Using container name: $name"

sudo docker run -it \
    -v /home/agrobenj/drake_workspace:/home/$USER/workspace \
    -p $port:22 \
    --name $name \
    drake_workspace \
    /bin/bash
