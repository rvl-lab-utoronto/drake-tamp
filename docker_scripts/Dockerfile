FROM ubuntu:18.04

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install python3-venv
RUN apt-get -y install ipython
# utilities 
RUN apt-get -y install vim
RUN apt-get -y install tmux
RUN apt-get -y install openssh-server
RUN apt-get install sudo
RUN apt-get -y install xfce4-terminal
RUN apt-get -y install gedit
RUN apt-get -y install trash-cli

RUN apt-get update
RUN curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz

RUN mkdir -p /python_venvs/tamp
RUN tar -xvzf drake.tar.gz -C /python_venvs/tamp --strip-components=1
RUN yes | /python_venvs/tamp/share/drake/setup/install_prereqs
WORKDIR /python_venvs/
RUN python3 -m venv tamp --system-site-packages
ENV PATH="/python_venvs/tamp/bin:$PATH"
WORKDIR /python_venvs/tamp

# ompl
RUN wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh -O /install-ompl-ubuntu.sh
RUN bash /install-ompl-ubuntu.sh --python

# pddlstream
RUN apt-get install -y git cmake g++ make
RUN git clone --recurse-submodules https://github.com/caelan/pddlstream.git pddlstream
RUN cd pddlstream && ./FastDownward/build.py release64
RUN cd pddlstream/FastDownward/builds && ln -s release64 release32
ENV PYTHONPATH=${PYTHONPATH}:/python_venvs/tamp/pddlstream

# other requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
