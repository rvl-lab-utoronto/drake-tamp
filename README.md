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

Note that building the python bindings for [ompl](https://github.com/ompl/ompl) and `RUN chown` will both take a long time.

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

Note: If you are trying to remote from a windows machine, the ProxyCommand is slightly different (see https://blog.because-security.com/t/how-to-use-ssh-proxycommands-on-windows-10-from-powershell/548 for more details):

```
Host tamp-workspace
  ProxyCommand ssh.exe -W %h:%p <remote-computer-host>
  HostName localhost 
  Port <port>
  User <your_username>
  LocalForward 5901 localhost:5901
```

## Folders and Files

- `docker_scripts` contains the files for building the docker image and starting the docker container.
- `learning` implements various models and data handling functions used in the `INFORMED` algorithm.
- `panda_station` contains code and urdf/sdf model files to build the simulation's in `Drake`. It is very similar to `Drake's` [ManipulationStation](https://github.com/RobotLocomotion/drake/blob/0995c9527dac6aa743b259319b26bf7249393b17/examples/manipulation_station/manipulation_station.cc)
- `pddlstream` is a fork of Caelan Garrett's [PDDLStream](https://github.com/caelan/pddlstream) with our implementation of INFORMED (see `pddlstream/algorithms/focused.py`)
- `experiments` contains scripts and files to run the TAMP experiments outlined in [LINK PAPER HERE]

## Experiments

### Top level scripts

#### main.py

Used to run a single trial.

Usage: 
``` bash
main.py [-h]
        [--domain {kitchen,blocks_world,two_arm_blocks_world,hanoi}]
        [--domain-options DOMAIN_OPTIONS] [--problem-file PROBLEM_FILE]
        [--oracle-options ORACLE_OPTIONS]
        [--algorithm {informedV2,adaptive}]
        [--mode {normal,save,oracle,complexityV3,complexityandstructure,complexitycollector,oracleexpansion,oraclemodel,model,cachingmodel,multiheadmodel,complexityoracle}]
        [--use-unique] [--should-save] [--max-time MAX_TIME]
        [--eager-mode] [--profile PROFILE] [--url URL]
        [--logpath LOGPATH] [--max_planner_time MAX_PLANNER_TIME]

optional arguments:
  -h, --help            show this help message and exit
  --domain {kitchen,blocks_world,two_arm_blocks_world,hanoi}
                        The name of the domain to test
  --domain-options DOMAIN_OPTIONS
                        Domain specific options in json kwarg format
  --problem-file PROBLEM_FILE
                        A path to the .yaml problem file
  --oracle-options ORACLE_OPTIONS
                        Keyword arguments passed to the model in json format,
                        like {"model_path":"/home/agrobenj/drake-
                        tamp/model_files/blocksworld_V2_adaptive/best.pt"}
  --algorithm {informedV2,adaptive}
                        Which algorithm do you want to run the trial with?
  --mode {normal,save,oracle,complexityV3,complexityandstructure,complexitycollector,oracleexpansion,oraclemodel,model,cachingmodel,multiheadmodel,complexityoracle}
  --use-unique          Make INFORMED use strictly unique results (refined)
                        during planning. Warning, this will usually drasically
                        slow things down
  --should-save         Force the oracle to save data to the save path
  --max-time MAX_TIME   Maximum total runtime of the trial
  --eager-mode          Do you want to run INFORMED in eager-mode? (i.e every
                        stream result popped of the queue has all it's
                        children added to the I^*
  --profile PROFILE     A path to (optionally) save a .profile file to (from
                        CProfile)
  --url URL             A meshcat url for viewing the problem
  --logpath LOGPATH     The directory to save the logs
  --max_planner_time MAX_PLANNER_TIME
                        The maximum time before FastDownward times out (per
                        call)
```

The various modes are as follows:

- normal: Run a normal trial
- save: Run a normal trial while saving information to `index.json` that an Oracle can later use to create labels
- oracle: Run a trial where the oracle is used to create labels. Another algorithm must have already been run, saving data to `index.json` or passing the path to `stats.json` directly as `--oracle-options='{"stats_path": "path/to/my/stats.json"}'`
- cachingmodel: Use a model that is faster because it caches previous computatinos (with INFORMED) 
- TODO: Explain the rest of these modes (the ones that are relevant)

#### run.sh

This script is used to run many experiments on a directory of problem files for the same domain and algorithm.

Usage:
`./run.sh <exp_dir> <problem_dir_path> <run_args1> <run_arg2> ...`

- `<exp_dir>`: A directory to save the output logs to (will be automatically created)
- `<problem_file_path>`: The directory containing the `.yaml` files specifying the problems (see `experiments/blocks_world/random/train/*`) for an example
- `<run_args{i}>`: The arguments that will be passed along to `main.py`

#### collect-labels.sh

This script is used to collect labels (in `learning/data/labeled/*`) used for training a model.

Usage:
`./collect-labels.sh <exp_dir> <domain> <problem_dir_path>`
- `<domain>`: The name of the domain to use. Run `python main.py -h` to see a list of valid domains (above).

#### start_meshcat_server.py

Run this python file to start a meshcat server. It will print out a url you can pass to `main.py` or any `run.py` to visualize the Drake simulation.

## Training

### Creating a dataset

After labels have been collected (see `collect-labels.sh`), we can create a dataset and train a model.
See example scripts `learning/scripts/{blocksworld.py, hanoi.py, kitchen_diffclasses.py}` for 
examples on how to create a `.json` file used to index a dataset. More information can be found in
`learning/gnn/data.py` under the `query_data` function, and the main block.

### Training and testing a model

`learning/gnn/main.py` is the top level script for training a model.

``` bash
Usage:
    main.py [-h] [--model-home MODEL_HOME] [--test-only]
        [--model {hyper,streamclass,streamclassv2}] [--epochs EPOCHS]
        [--save-every SAVE_EVERY] [--pos-weight POS_WEIGHT]
        [--stratify-train-prop STRATIFY_TRAIN_PROP] [--lr LR]
        [--gradient-batch-size GRADIENT_BATCH_SIZE]
        [--batch-size BATCH_SIZE]
        [--num-preprocessors NUM_PREPROCESSORS] [--from-best]
        [--use-problem-graph] [--datafile DATAFILE] [--debug]
        [--epoch-size EPOCH_SIZE] [--preprocess-all] [--ablation]

optional arguments:
  -h, --help            show this help message and exit
  --model-home MODEL_HOME
                        A directory in which logs and parameters will be saved
                        during training
  --test-only           If you only want to test a model (not end-to-end)
  --model {hyper,streamclass,streamclassv2}
                        The type of model you want to train. See
                        learning/gnn/models.py for more information
  --epochs EPOCHS       The number of epochs to train for
  --save-every SAVE_EVERY
                        The number of epochs between tests on the validation
                        set and saving model parameters
  --pos-weight POS_WEIGHT
                        The weighting given to positive examples in the
                        training set
  --stratify-train-prop STRATIFY_TRAIN_PROP
                        The proportion of positive examples shown to the model
  --lr LR               The learning rate
  --gradient-batch-size GRADIENT_BATCH_SIZE
  --batch-size BATCH_SIZE
                        The batch size
  --num-preprocessors NUM_PREPROCESSORS
                        The number of cpu cores allowed to help preprocess the
                        data
  --from-best           Do you want to use the best.pt file saved in the model
                        directory during testing
  --use-problem-graph   Do you want to use the problem graph model
  --datafile DATAFILE   A path to a json file containing train and validation
                        keys wich have a list of pkl paths as values.
  --debug               Are you debugging?
  --epoch-size EPOCH_SIZE
                        The number of labels shown per epoch
  --preprocess-all      Do you want to preprocess all of the data, or
                        processes it when it is needed?
  --ablation            Are you doing an ablation study?
```
