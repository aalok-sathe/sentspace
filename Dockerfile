# syntax=docker/dockerfile:1
# FROM python:3.8-alpine
FROM ubuntu:latest

# install necessary packages using apk
RUN apt update
RUN apt install -y python3.8 python3.8-dev python3-pip
RUN apt install -y python2.7 python2.7-dev 
RUN apt install -y openjdk-8-jdk-headless openjdk-8-jre-headless
RUN apt install -y git build-essential
# clear cache once we're done installing stuff

WORKDIR /sentspace/
COPY . /sentspace/ 
# ADD . /sentspace/

# install necessary packages using pip
RUN python3.8 -m pip install -U pip

RUN git clone https://github.com/njsmith/zs.git
RUN python3.8 -m pip install cython
RUN cd zs && python3.8 -m pip install .
RUN rm -rf ./zs

RUN python3.8 -m pip install torch==1.9.0 torchvision torchaudio
RUN python3.8 -m pip install -r ./requirements.txt

# remove packages we won't need at runtime
RUN apt remove git

RUN python3.8 -m pip cache purge
RUN apt clean && rm -rf /var/lib/apt/lists/*

# RUN yarn install --production
# CMD ["node", "src/index.js"]