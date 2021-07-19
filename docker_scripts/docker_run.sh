#!/bin/bash

echo ""
echo "Running docker-tamp-$USER container"
echo "You can specify the port with the -p flag and the container name with the -n flag"
echo ""

port=2300
name=drake-tamp-$USER

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
    -v /home/$USER/drake-tamp:/home/$USER/drake-tamp \
    --shm-size 8G \
    --gpus all \
    -p $port:22 \
    --name $name \
    drake-tamp-$USER \
    /bin/bash
