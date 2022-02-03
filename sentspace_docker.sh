#/usr/bin/env bash
# set -x
set -e
sudo docker run \
  --name sentspace \
  --mount type=bind,source="$(pwd)",target=/app/workdir \
  --rm \
  aloxatel/sentspace:latest "$@"
