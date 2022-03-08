FROM ubuntu:16.04

RUN apt-get update && \
    apt-get -y install sudo && \
    sudo apt-get update -qq && \
    apt-get install -y python3 \
                       python3-pip \
                       python3-dev \
                       build-essential \
                       software-properties-common \
                       openjdk-8-jdk \
                       git \
                       wget && \
    sudo add-apt-repository ppa:openjdk-r/ppa && \
         apt-get update -qq && \
         apt-get install -y openjdk-8-jdk

RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home
ENV JAVA_HOME=/docker-java-home \
    JCC_JDK=/docker-java-home

RUN apt-get install libffi-dev && \
    python3 -m pip install --upgrade "pip < 21.0" \
                                     wheel \
                                     JCC \
                                     urllib3 && \
    python3 -m pip install jupyter \
                           nilearn \
                           sklearn \
                           nose \
                           matplotlib \
                           scipy \
                           psutil

RUN useradd --no-user-group --create-home --shell /bin/bash neuro && \
    mkdir /home/neuro/nighres
COPY build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE imcntk-lib-files.sh /home/neuro/nighres/
COPY nighres /home/neuro/nighres/nighres

RUN cd /home/neuro/nighres && \
    ./build.sh && \
    cd /home/neuro/nighres && python3 -m pip install . && \
    mkdir /home/neuro/notebooks && \
    chown -R neuro /home/neuro

COPY docker/jupyter_notebook_config.py /etc/jupyter/

EXPOSE 8888

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

USER neuro
