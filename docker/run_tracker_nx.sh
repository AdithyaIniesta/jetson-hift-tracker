#!/usr/bin/env bash
docker run --rm -it --runtime nvidia \
  --network host \
  --device /dev/video0 \
  --device /dev/ttyTHS0 \
  jvp/jetsontracker:v2.0-jp5-orin-xavier \
  /opt/jvp/run_docker.sh
