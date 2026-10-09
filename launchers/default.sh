#!/bin/bash

source /environment.sh

dt-launchfile-init

dt-exec python3 "$DT_REPO_PATH/packages/quackcomm/scripts/sensor_bridge.py"

dt-launchfile-join
