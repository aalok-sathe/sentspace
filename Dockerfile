# syntax=docker/dockerfile:1
FROM ubuntu:latest

# install necessary packages using apt
RUN apt update
RUN apt install -y python3.8 python3.8-dev python3-pip
RUN apt install -y python2.7 python2.7-dev 
# RUN apt install -y openjdk-8-jdk-headless openjdk-8-jre-headless
RUN apt install -y build-essential git
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt install -y python3-icu

WORKDIR /sentspaceapp/
COPY ./requirements.txt /sentspaceapp/requirements.txt 
# ADD . /sentspace/

# install ZS package separately (pypi install fails)
RUN python3.8 -m pip install -U pip cython
RUN git clone https://github.com/njsmith/zs
RUN cd zs && git checkout v0.10.0 && pip install .
RUN rm -rf zs
# install rest of the requirements using pip
RUN pip install -r ./requirements.txt
RUN polyglot download morph2.en

# cleanup
RUN apt remove -y git
RUN apt autoremove -y
RUN pip cache purge
RUN apt clean && rm -rf /var/lib/apt/lists/*

# RUN yarn install --production
# CMD ["node", "src/index.js"]