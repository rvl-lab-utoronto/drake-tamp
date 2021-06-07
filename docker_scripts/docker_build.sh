#!/bin/bash


echo ""
echo "Building image drake-tamp-$USER"

if [[ ! $1 ]]; then
    echo "You need to specify as password for this image, like so: sudo docker_build.sh my_password"
    echo "exiting build script"
    exit 1
fi

if [[ ! $USER ]]; then
    echo "You USER environment variable is not set"
    echo "exiting build script"
    exit 1
fi

if [[ ! $UID  ]]; then
    echo "You UID environment variable is not set"
    echo "exiting build script"
    exit 1
fi

echo "root password: $1"
echo "user: $USER"
echo "id: $UID"
echo ""

sudo docker image build --build-arg password=$1 --build-arg user=$USER --build-arg id=$UID -t drake-tamp-$USER .
