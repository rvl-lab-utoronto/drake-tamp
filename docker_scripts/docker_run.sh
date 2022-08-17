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

xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge 

sudo docker run -it \
    -v /home/$USER/drake-tamp:/home/$USER/drake-tamp \
    --privileged \
    --shm-size 8G \
    --gpus all \
    -p $port:22 \
    -e DISPLAY=$DISPLAY \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    -e XAUTHORITY=$XAUTH \
    --name $name \
    drake-tamp-$USER \
    /bin/bash
