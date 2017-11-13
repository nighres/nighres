FROM ubuntu:14.04
RUN sudo apt-get update -qq && apt-get install -y python python-pip python-dev build-essential software-properties-common
RUN sudo add-apt-repository ppa:openjdk-r/ppa && apt-get update -qq && apt-get install -y openjdk-8-jdk
RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home
ENV JAVA_HOME /docker-java-home
ENV JCC_JDK /docker-java-home

RUN sudo apt-get install -y git python-pip python-dev wget jcc

RUN useradd -g root --create-home --shell /bin/bash neuro \
    && usermod -aG sudo neuro \
    && usermod -aG users neuro

RUN pip install --upgrade wheel JCC twine urllib3 pip 
RUN mkdir /home/neuro/nighres
COPY build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE /home/neuro/nighres/
COPY nighres /home/neuro/nighres/nighres
RUN cd /home/neuro/nighres && ./build.sh
RUN cd /home/neuro/nighres && pip install .

RUN pip install jupyter nilearn sklearn nose matplotlib
COPY docker/jupyter_notebook_config.py /etc/jupyter/

RUN mkdir /home/neuro/notebooks
RUN chown -R neuro /home/neuro

EXPOSE 8888

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

USER neuro
