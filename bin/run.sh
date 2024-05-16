#!/bin/bash

# Get the directory of the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "${SCRIPT_DIR}")

# Load the environment variables
source "${PARENT_DIR}/.env"

CONTAINER_NAME=${MODEL_APP_CONTAINER_NAME}
IMAGE_NAME=${MODEL_APP_IMAGE_NAME}

echo "Starting a new container ${CONTAINER_NAME}..."
docker run -p 8000:8000 \
    --name ${CONTAINER_NAME} \
    --rm \
    ${IMAGE_NAME}
