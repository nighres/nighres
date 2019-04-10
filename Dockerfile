FROM ubuntu:14.04

# install linux and java dependencies
RUN sudo apt-get update -qq && apt-get install -y \
    git \
    jcc \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    software-properties-common \
    wget
RUN sudo add-apt-repository ppa:openjdk-r/ppa && apt-get update -qq && apt-get install -y openjdk-8-jdk
RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home
ENV JAVA_HOME /docker-java-home
ENV JCC_JDK /docker-java-home
RUN useradd --no-user-group --create-home --shell /bin/bash neuro

# Install Tini
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Add Python dependencies
RUN mkdir /home/neuro/nighres
WORKDIR /home/neuro/nighres
COPY ./requirements.txt .
COPY requirements/base.txt ./requirements/
# Install Python dependencies
RUN python3 -m pip install --upgrade wheel JCC twine urllib3 pip
RUN python3 -m pip install -r requirements.txt

# CBS Tools installation files
COPY build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE imcntk-lib-files.sh /home/neuro/nighres/
COPY nighres /home/neuro/nighres/nighres
COPY docker/jupyter_notebook_config.py /etc/jupyter/
# Run CBS Tools installation
RUN cd /home/neuro/nighres && ./build.sh
RUN cd /home/neuro/nighres && python3 -m pip install .

RUN chown -R neuro /home/neuro

EXPOSE 8888
USER neuro

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
