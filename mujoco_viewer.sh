#!/bin/bash

DEFAULT_FILE="./genesis/assets/xml/stryon_no3/stryon_no3.xml"
MJCF_FILE="${1:-$DEFAULT_FILE}"
python -m mujoco.viewer --mjcf="$MJCF_FILE"
