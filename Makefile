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
		--copy build.sh cbstools-lib-files.sh setup.py MANIFEST.in README.rst LICENSE imcntk-lib-files.sh /home/neuro/nighres/ \
		--copy nighres /home/neuro/nighres/nighres \
		--workdir /home/neuro/nighres \
		--run "conda init && . /root/.bashrc && conda activate nighres && conda info --envs && ./build.sh && rm -rf cbstools-public imcn-imaging nighresjava/build nighresjava/src" \
		--miniconda \
			use_env="nighres" \
			conda_install="jupyter" \
			pip_install="." \
		--copy docker/jupyter_notebook_config.py /etc/jupyter \
		--expose 8888 \
		--run 'chmod -R 777 /home/neuro/' \
		--user neuro \
		--cmd "jupyter notebook --no-browser --ip 0.0.0.0" > Dockerfile

docker_build: Dockerfile
	docker build . -t nighres:latest

docker_run: 
	mkdir -p $$PWD/examples
	docker run -it --rm \
		--publish 8888:8888 \
		--volume $$PWD/examples:/home/neuro/nighres/examples \
		nighres:latest \
			jupyter-notebook --no-browser --ip 0.0.0.0
docker_test: docker_test_cortical_depth_estimation docker_test_coregistration

docker_test_classification: docker_build
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres/nighres_examples \
		nighres:latest \
			python3 /examples/example_01_tissue_classification.py 

docker_test_cortical_depth_estimation: docker_test_classification
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres/nighres_examples \
		nighres:latest \
			python3 /examples/example_02_cortical_depth_estimation.py 

docker_test_coregistration:
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres/nighres_examples \
		nighres:latest \
			python3 /examples/example_03_brain_coregistration.py