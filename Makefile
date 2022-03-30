.PHONY: clean install

clean:
	rm -rf nighresjava/
	rm -rf cbstools-public/
	rm -rf imcn-imaging/
	rm -rf nighres.egg-info/
	rm -rf nighres_examples/

install:
	./build.sh
	pip install .

# =============================================================================
# Docker related content

.PHONY: Dockerfile docker_build

Dockerfile:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base debian:stretch-slim \
		--pkg-manager apt \
		--install "openjdk-8-jdk git wget build-essential software-properties-common libffi-dev" \
		--miniconda \
			create_env="nighres" \
			conda_install="python=3.9 pip jcc gcc_linux-64 gxx_linux-64 Nilearn" \
			activate="true" \
		--env JAVA_HOME="/docker-java-home" \
		--env JCC_JDK="/docker-java-home" \
		--run 'ln -svT "/usr/lib/jvm/java-8-openjdk-$$(dpkg --print-architecture)" /docker-java-home' \
		--copy Makefile build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE imcntk-lib-files.sh /home/neuro/nighres/ \
		--copy nighres /home/neuro/nighres/nighres \
		--workdir /home/neuro/nighres \
		--run "conda init && . /root/.bashrc && conda activate nighres && conda info --envs && ./build.sh" \
		--miniconda \
			use_env="nighres" \
			conda_install="jupyter" \
			pip_install="." \
		--copy docker/jupyter_notebook_config.py /etc/jupyter \
		--expose 8888 \
		--cmd "jupyter lab --port=8888 --no-browser --ip=0.0.0.0" \
		--run 'rm -rf cbstools-public imcn-imaging nighresjava/build nighresjava/src' \
		--user neuro > Dockerfile

docker_build: Dockerfile
	docker build . -t nighres:latest
