# Drake-TAMP 

Task and motion planning in [Drake](https://drake.mit.edu/)

## Installation and Setup

Prerequisites:
- [Docker](https://docs.docker.com/get-docker/). The code can also be run outside of a container. See the nessecary packages in `docker_scripts/Dockerfile` and `docker_scripts/requirements.txt`
- make sure your environment variables `$USER` and `$UID` are set

Clone the repository and build the docker image. Fill in `<your_password>` with the login password you want for your user in the docker container:

```
cd ~
git clone --recurse-submodules https://github.com/rvl-lab-utoronto/drake-tamp.git
cd ~/drake-tamp/docker_scripts/
./docker_build.sh <your_password>
```

To compile FastDownward run:

```
cd pddlstream && ./FastDownward/build.py release64
cd pddlstream/FastDownward/builds && ln -s release64 release32
```

Start the container, specifying the port you want to use for ssh `<port>` and the container name `<container_name>`:

```
cd ~/drake-tamp/docker_scripts/
./docker_run.sh -p <port> -n <container_name>
```

If you are working on a remote machine, it is nice to ssh into the docker container and use a remote desktop  so you can use all GUIs. To do so, on your local machine, add this to your ~/.ssh/config file:

```
Host tamp-workspace
    ProxyCommand ssh -q <remote-computer-hostname> -q0 localhost <port>
    LocalForward 5901 localhost:5901
    User <your_username>
```

## Folders and Files

`docker_scripts` contains the files for building the docker image and starting the docker container.

TODO(agro): fill out readme


