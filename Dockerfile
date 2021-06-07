FROM ubuntu:18.04

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install python3-venv
RUN apt-get -y install ipython

RUN mkdir -p /python_venvs/tamp
WORKDIR /python_venvs/
RUN python3 -m venv tamp --system-site-packages
ENV PATH="/python_venvs/tamp/bin:$PATH"
WORKDIR /python_venvs/tamp

# pddlstream
RUN apt-get install -y git cmake g++ make
RUN git clone --recurse-submodules https://github.com/caelan/pddlstream.git pddlstream
RUN cd pddlstream && ./FastDownward/build.py release64
RUN cd pddlstream/FastDownward/builds && ln -s release64 release32
ENV PYTHONPATH=${PYTHONPATH}:/python_venvs/tamp/pddlstream

# other requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
