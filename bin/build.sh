#!/bin/bash

# Get the directory of the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "${SCRIPT_DIR}")

# Load the environment variables
source "${PARENT_DIR}/.env"

docker build --no-cache -t ${MODEL_APP_IMAGE_NAME} ${PARENT_DIR}