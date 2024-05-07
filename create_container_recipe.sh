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

docker run ${expose} \
    --rm repronim/neurodocker:0.9.5 generate "${container_type}" \
    --base-image debian:stable \
    --pkg-manager apt \
    --install "openjdk-17-jdk git wget build-essential software-properties-common libffi-dev" \
    --env JAVA_HOME="/docker-java-temurin-home" \
    --env JCC_JDK="/docker-java-temurin-home" \
    --run 'ln -svT "/usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture)" /docker-java-temurin-home' \
    --miniconda \
        version="latest" \
        env_name="nighres" \
        env_exists="false" \
        conda_install="python=3.9 pip JCC Nilearn" \
    --copy build.sh cbstools-lib-files.sh imcntk-lib-files.sh dependencies_sha.sh setup.py setup.cfg MANIFEST.in README.rst LICENSE /nighres/ \
    --copy nighres /nighres/nighres \
    --workdir /nighres \
    --run "conda init && . /root/.bashrc && activate nighres && conda info --envs && pip install jcc && ./build.sh && rm -rf cbstools-public imcn-imaging nighresjava/build nighresjava/src" \
    --miniconda \
        version="latest" \
        installed="true" \
        env_name="nighres" \
        conda_install="jupyterlab" \
        pip_install="." \
    --copy docker/jupyter_notebook_config.py /etc/jupyter \
    --user neuro \
    --workdir /home/neuro >"${output_file}"

# Needed as the copy command does not work (yet?) in neurodocker for singularity 
if [ "${container_type}" == "singularity" ]; then

    files_to_copy="build.sh cbstools-lib-files.sh setup.py setup.cfg MANIFEST.in README.rst LICENSE imcntk-lib-files.sh dependencies_sha.sh"

    dest_folder='\/nighres\/'

    str="%files\n"
    for file in ${files_to_copy}; do
        str="${str}${file} ${dest_folder}\n"
    done

    str="${str}nighres ${dest_folder}nighres\/\n"
    str="${str}docker\/jupyter_notebook_config.py \/etc\/jupyter\/\n"
    replace="${str}\n%post"

    search="%post"
    sed -i "s/$search/$replace/" $output_file

fi
