FROM ubuntu:15.10

# Basic docker parsa image: build-essentials, ssh and a parsa user
RUN apt-get update && apt-get install -y build-essential checkinstall wget python-pip sudo python-dev openssh-server \
             swig software-properties-common openjdk-8-jdk pkg-config python-numpy zip g++ zlib1g-dev unzip git-all \
	     emacs24-nox python3-numpy python3-dev python3-setuptools screen curl && \
     mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    echo 'ALL: LOCAL' > /etc/ssh/hosts.allow && \
    sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    useradd -d /home/parsa -m parsa && \
    adduser parsa sudo && \
    echo 'parsa:parsa' | chpasswd && \
    easy_install3 pip && \
    pip3 install wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm /bin/sh && ln -s /bin/bash /bin/sh && \
    sudo update-ca-certificates -f

USER parsa
ENV HOME /home/parsa

RUN git config --global user.email "mario.drumond@epfl.ch" && \
    git config --global user.name "Mario Paulo Drumond" && \
    cd ~/ && \
    git clone https://github.com/StateFarmIns/bazel.git && \
    cd bazel && \
    ./compile.sh && \
    pip3 install --upgrade virtualenv --user
    echo 'export PATH=/home/parsa/bazel/output:/home/parsa/.local/bin:$PATH' >> $HOME/.bashrc && \
    cd ~/ && \
    git clone --recurse-submodules https://github.com/mdrumond/tensorflow.git && \
    cd ~/tensorflow && \
    ./get_deps.sh

COPY .emacs /home/parsa/.emacs

USER root
CMD ["/usr/sbin/sshd", "-D"]