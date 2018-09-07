FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y \
    python3 python3-nose python3-pandas python-h5py \
    python python-numpy python-nose python-pandas python3-h5py \
    pep8 python-pip python3-pip python-wheel \
    python-sphinx && \
    pip3 install \
    --upgrade pip \
    --upgrade setuptools \
    numpy \
    scipy \
    scikit-learn==0.19.2 \
    sklearn