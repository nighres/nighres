#!/usr/bin/env bash
# set -e -x

# script to call neurodocker to create container recipes
# 
#  USAGE:
# 
# bash create_container_recipe.sh docker
# 
# bash create_container_recipe.sh singularity

container_type="${1:-docker}"

expose=""

if [ "${container_type}" == "docker" ]; then
    output_file="Dockerfile"
    expose="--expose 8888"

elif [ "${container_type}" == "singularity" ]; then
    output_file="NighresSingularity.def"

else
    echo "Unknown container type: ${container_type}. Must be 'docker' or 'singularity'."
    exit 1

fi

docker run --rm repronim/neurodocker:0.7.0 generate "${container_type}" \
    --base debian:stretch-slim \
    --pkg-manager apt \
    --install "openjdk-8-jdk git wget build-essential software-properties-common libffi-dev" \
    --miniconda \
        create_env="nighres" \
        conda_install="python=3.9 pip jcc Nilearn" \
        activate="true" \
    --env JAVA_HOME="/docker-java-home" \
    --env JCC_JDK="/docker-java-home" \
    --run 'ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home' \
    --copy build.sh cbstools-lib-files.sh imcntk-lib-files.sh dependencies_sha.sh setup.py setup.cfg MANIFEST.in README.rst LICENSE /nighres/ \
    --copy nighres /nighres/nighres \
    --workdir /nighres \
    --run "conda init && . /root/.bashrc && conda activate nighres && conda info --envs && ./build.sh && rm -rf cbstools-public imcn-imaging nighresjava/build nighresjava/src" \
    --miniconda \
        use_env="nighres" \
        conda_install="jupyterlab" \
        pip_install="." \
    --copy docker/jupyter_notebook_config.py /etc/jupyter \
    "${expose}" \
    --user neuro \
    --workdir /home/neuro >"${output_file}"
