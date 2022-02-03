# syntax=docker/dockerfile:1
FROM ubuntu:20.04


################################################################
#### set up environment ####
####  source: https://stackoverflow.com/a/54763270/2434875
####  source: https://github.com/pypa/pip/issues/5735
################################################################
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.10

WORKDIR /app


################################################################
#### install system-wide dependencies and utilities; cache ####
################################################################
RUN apt update
RUN apt install -y python3.8 python3.8-dev python3-pip
RUN apt install -y python2.7 python2.7-dev 
# RUN apt install -y openjdk-8-jdk-headless openjdk-8-jre-headless
RUN apt install -y build-essential curl 
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt install -y pkg-config libicu-dev python3-icu

# RUN pip install "poetry==$POETRY_VERSION"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.10/get-poetry.py | python3.8 -
RUN ln -s "$HOME/.poetry/bin/poetry" /usr/bin/poetry
COPY poetry.lock pyproject.toml /app/
COPY sentspace /app/sentspace
# RUN python3.8 -m venv /venv


################################################################
#### install package dependencies; build using poetry ####
################################################################
RUN ls -lah .
RUN pip config set global.cache-dir false
RUN poetry config virtualenvs.create false 
RUN poetry install -E polyglot --no-interaction --no-ansi --no-root && \
    poetry build -f wheel && \
    pip install --no-deps . dist/*.whl && \
    rm -rf dist *.egg-info

RUN polyglot download morph2.en


################################################################
#### cleanup ####
################################################################
RUN apt remove -y git
RUN apt autoremove -y
RUN apt clean && rm -rf /var/lib/apt/lists/*


################################################################
#### set up entrypoint to use as standalone app ####
################################################################
EXPOSE 8051
ENTRYPOINT [ "python3.8", "-m", "sentspace" ]
CMD [ "-h" ]