#/usr/bin/env bash
# set -x
set -e
sudo docker run \
  --name sentspace \
  --mount type=bind,source="$(pwd)",target=/app/workdir \
  --net=host \
  --rm \
  aloxatel/sentspace:latest "$@"
