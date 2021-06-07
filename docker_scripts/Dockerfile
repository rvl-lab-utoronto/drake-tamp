FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install wget unzip 

RUN apt-get -y update
RUN apt-get -y install git cmake g++ make

# for display through vnc
RUN apt-get -y update
RUN apt-get install -y wget unzip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y xfce4 tigervnc-standalone-server

RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install python3-venv
RUN apt-get -y install ipython

# utilities 
RUN apt-get update
RUN apt-get -y install firefox
RUN apt-get -y install vim
RUN apt-get -y install tmux
RUN apt-get -y install openssh-server
RUN apt-get -y install sudo
RUN apt-get -y install xfce4-terminal
RUN apt-get -y install gedit
RUN apt-get -y install trash-cli

# drake
RUN apt-get update
RUN curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz

# python venv setup with drake
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
RUN git clone --recurse-submodules https://github.com/caelan/pddlstream.git pddlstream
RUN cd pddlstream && ./FastDownward/build.py release64
RUN cd pddlstream/FastDownward/builds && ln -s release64 release32
ENV PYTHONPATH=${PYTHONPATH}:/python_venvs/tamp/pddlstream

# update terminal config
RUN printf "2\n" | update-alternatives --config x-terminal-emulator

# arguments when building image
# login password 
ARG password 
# $UID
ARG id
# $USER
ARG user
ENV PASS=$password

RUN groupadd -g $id $user
RUN useradd -s /bin/bash -u $id -g $id -m $user && echo "$user:$password" | chpasswd && adduser $user sudo
RUN chown -R $user /python_venvs 
USER $user

# other requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN echo "source /python_venvs/tamp/bin/activate" >> ~/.bashrc
RUN echo "export  $USER=$user" >> ~/.bashrc

WORKDIR /home/$user/drake-tamp
COPY .tmux.conf /home/$user/.tmux.conf
COPY .vimrc /home/$user/.vimrc

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]

CMD ["bash"]
