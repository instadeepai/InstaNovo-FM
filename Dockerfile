ARG UV_DIR=/usr/local/bin/uv
ARG UV_PYTHON_INSTALL_DIR=/python
ARG UV_PROJECT_ENVIRONMENT=/app
ARG UV_VERSION=0.6.9
ARG UV_PYTHON=python3.11.9
ARG UV_COMPILE_BYTECODE=1
ARG UV_PYTHON_DOWNLOADS=never

# Stage: 'uv'
# It is used to define the uv Docker image
FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

# Stage: 'env'
# It is used to define python version and install all the Python dependencies
FROM ubuntu:noble AS env

# Copy uv with proper version
ARG UV_DIR
COPY --from=uv /uv $UV_DIR

# Set location for python installed by uv
ARG UV_PYTHON_INSTALL_DIR
ENV UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR

# Set /app for the virtual environment created by uv
ARG UV_PROJECT_ENVIRONMENT
ENV UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

# Define Python version
ARG UV_PYTHON
ENV UV_PYTHON=$UV_PYTHON

# Byte-compile the Python files for faster application startup
ARG UV_COMPILE_BYTECODE
ENV UV_COMPILE_BYTECODE=$UV_COMPILE_BYTECODE

# Install the required system dependencies
# Clean after packages' install
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install curl git apt-transport-https ca-certificates  -y && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the files with locked dependencies
COPY pyproject.toml /tmp/pyproject.toml
COPY uv.lock /tmp/uv.lock

# Install all the dependencies with uv
RUN cd /tmp && uv sync --locked --no-dev --no-cache --no-install-project

# Stage: 'runtime'
# It is used to define the image for the runtime with:
#   - linux packages
#   - non root user
#   - python packages
#   - env variables
# Fix the hash of the base image as the latest pushed image under this tag is broken
FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04@sha256:1c3cefb97f774264b9709eb209aa3a910ce5ba56889aa85d19d525a29ac01523 AS runtime

# Use default values for the CI
ARG HOST_UID=42000
ARG HOST_GID=42001

ENV USER=appuser
ENV HOME_DIRECTORY=/home/$USER/mass_spectrometry_foundation_model

# Do not create .pyc file cf https://stackoverflow.com/a/60797635/8056572
ENV PYTHONDONTWRITEBYTECODE=1

# Allow to have log in real time
ENV PYTHONUNBUFFERED=1

# Use same GPU IDs between nvidia-smi and tensorflow
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Do not use all the GPU memory by default
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Remove the tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3

# Install the required system dependencies
# Clean after packages' install
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install curl git -y && \
    rm -rf /var/lib/apt/lists/*

# Create group and user, add -f to skip the command without error if it exists already
RUN groupadd --force --gid $HOST_GID $USER && \
        useradd -r -m --uid $HOST_UID --gid $HOST_GID $USER

# Ensure there is no prompt for password when running sudo
RUN echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the default user
USER $USER

# Set the PATH to have access to python executable
ARG UV_PROJECT_ENVIRONMENT
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Set /app for the virtual environment created by uv
ENV UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Set PYTHONPATH to access the 'mass_spectrometry_foundation_model' Python package in the current directory
ENV PYTHONPATH="."

# Set the terminal color
ENV TERM=xterm-256color

# Copy uv, python and the installed packages
ARG UV_DIR
ARG UV_PYTHON_INSTALL_DIR
ARG UV_PROJECT_ENVIRONMENT
COPY --chown=$USER:$USER --from=env $UV_DIR $UV_DIR
COPY --chown=$USER:$USER --from=env $UV_PYTHON_INSTALL_DIR $UV_PYTHON_INSTALL_DIR
COPY --chown=$USER:$USER --from=env $UV_PROJECT_ENVIRONMENT $UV_PROJECT_ENVIRONMENT

# Stage: dev
# It contains all the dependencies to run the Python package and its associated tests with pytest
FROM runtime AS dev

# Prevent uv from downloading isolated Python builds as Python is already available
ARG UV_PYTHON_DOWNLOADS
ENV UV_PYTHON_DOWNLOADS=$UV_PYTHON_DOWNLOADS

# Byte-compile the python files for faster application startup
ARG UV_COMPILE_BYTECODE
ENV UV_COMPILE_BYTECODE=$UV_COMPILE_BYTECODE

# Copy the files with locked dependencies
COPY --chown=$USER:$USER pyproject.toml /tmp/pyproject.toml
COPY --chown=$USER:$USER uv.lock /tmp/uv.lock

# Synchronize dependencies to also include dev-specific dependencies
RUN cd /tmp && uv sync --locked --no-cache --no-install-project && rm /tmp/pyproject.toml /tmp/uv.lock

# Stage: 'aichor'
# It is used to run the Python package on AIchor which requires to have the package installed
FROM runtime AS aichor

# Install the 'mass_spectrometry_foundation_model' package
COPY --chown=$USER . .
RUN uv pip install ./dreams
