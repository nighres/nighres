FROM ubuntu:14.04
RUN sudo apt-get update -qq && apt-get install -y python3 python3-pip python3-dev build-essential software-properties-common
RUN sudo add-apt-repository ppa:openjdk-r/ppa && apt-get update -qq && apt-get install -y openjdk-8-jdk
RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home
ENV JAVA_HOME /docker-java-home
ENV JCC_JDK /docker-java-home

RUN sudo apt-get install -y git python3-pip python3-dev wget jcc

RUN useradd --no-user-group --create-home --shell /bin/bash neuro

RUN python3 -m pip install --upgrade wheel JCC twine urllib3
RUN mkdir /home/neuro/nighres
COPY build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE imcntk-lib-files.sh /home/neuro/nighres/
COPY nighres /home/neuro/nighres/nighres

RUN python3 -m pip install --upgrade pip
RUN cd /home/neuro/nighres && ./build.sh
RUN cd /home/neuro/nighres && python3 -m pip install .

RUN python3 -m pip install jupyter nilearn sklearn nose matplotlib scipy
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

RUN python3 -m pip install psutil

USER neuro

