# syntax=docker/dockerfile:1
FROM ubuntu:20.04

# source: https://stackoverflow.com/a/54763270/2434875
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.10

WORKDIR /app
# install necessary packages using apt
RUN apt update
RUN apt install -y python3.8 python3.8-dev python3-pip
RUN apt install -y python2.7 python2.7-dev 
# RUN apt install -y openjdk-8-jdk-headless openjdk-8-jre-headless
RUN apt install -y build-essential curl
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt install -y python3-icu

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.10/get-poetry.py | python3.8 -
# RUN pip install "poetry==$POETRY_VERSION"

ADD poetry.lock pyproject.toml /app/
# RUN python3.8 -m venv /venv
RUN poetry config virtualenvs.create false 
RUN poetry install -E polyglot --no-interaction --no-ansi --no-root && \
    poetry build -f wheel && \
    pip install --no-deps . dist/*.whl && \
    rm -rf dist *.egg-info
# ADD ./requirements.txt /app/requirements/requirements.txt 
# unnecessary: # ADD . /app/

# install ZS package separately (pypi install fails)
# RUN python3.8 -m pip install -U pip cython
# RUN apt install -y git
# RUN git clone https://github.com/njsmith/zs
# RUN cd zs && git checkout v0.10.0 && pip install .
# RUN rm -rf zs
# install rest of the requirements using pip
# RUN pip install -r ./requirements.txt

RUN polyglot download morph2.en
# RUN pip install -U ipython ipykernel jupyter

# cleanup
RUN apt remove -y git
RUN apt autoremove -y
RUN pip cache purge
RUN apt clean && rm -rf /var/lib/apt/lists/*

EXPOSE 8051
ENTRYPOINT [ "python3.8", "-m", "sentspace" ]
CMD [ "-h" ]