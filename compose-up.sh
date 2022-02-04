#!/bin/bash

set -a # automatically export all variables

COMMIT=$(git rev-parse --short HEAD)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

COMPOSE_PROJECT_NAME=carla-pedestrians
PLATFORM=nvidia  # nvidia or cpu
SHM_SIZE=8gb

COMMON_DIR=./pedestrians-common
CARLA_SERVER_DIR=${COMMON_DIR}/server
VIDEO2CARLA_DIR=./pedestrians-video-2-carla
SCENARIOS_DIR=./pedestrians-scenarios
VIZ_DIR=./carlaviz

source ${COMMON_DIR}/.env
source ${VIDEO2CARLA_DIR}/.env
source ${SCENARIOS_DIR}/.env
source ${VIZ_DIR}/.env

# Use BuildKit by default
COMPOSE_DOCKER_CLI_BUILD=1
DOCKER_BUILDKIT=1 

set +a # end of automatic export

if [ $PLATFORM == "cpu" ]; then
    COMPOSE_ARGS=(-f "${VIDEO2CARLA_DIR}/docker-compose.yml"
                  -f "${VIDEO2CARLA_DIR}/docker-compose.cpu.yml")
else
    COMPOSE_ARGS=(-f "${CARLA_SERVER_DIR}/docker-compose.yml"
                  -f "${VIDEO2CARLA_DIR}/docker-compose.yml"
                  -f "${VIZ_DIR}/docker-compose.yml"
                  -f "${SCENARIOS_DIR}/docker-compose.yml")
fi

# first, build the common image used by the containers
# specifying the docker-compose.yml from the main repo first allows to set the 'root path' correctly
docker-compose \
    -f "docker-compose.yml" \
    -f "${COMMON_DIR}/docker-compose.yml" \
    build

# then build & run the actual services
# selected services can be run by passing their names as arguments
docker-compose \
    -f "docker-compose.yml" \
    ${COMPOSE_ARGS[@]} \
    up -d --build $@

# Display some info for the user about carlaviz
echo "To access carlaviz, open a browser and go to http://localhost:8080."
