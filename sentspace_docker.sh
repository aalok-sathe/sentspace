#/usr/bin/env bash
# set -x
sudo docker run \
  --name sentspace \
  --mount type=bind,source="$(pwd)",target=/app/workdir \
  --rm \
  aloxatel/sentspace:latest "$@"
