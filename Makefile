.PHONY: clean install clean_env_file

clean:
	rm -rf nighresjava/
	rm -rf cbstools-public/
	rm -rf imcn-imaging/
	rm -rf nighres.egg-info/
	rm -rf nighres_examples/

install:
	./build.sh
	pip install .

clean_env_file: conda-nighres.yml
	sed -i '/^.*nighres.*/d' ./conda-nighres.yml
	sed -i '/^prefix:.*/d' ./conda-nighres.yml

update_dep_shasum:
	cbstools_sha=$$(git ls-remote https://github.com/piloubazin/cbstools-public.git master | head -n 1 | cut -c1-7) && \
		echo cbstools_sha=$${cbstools_sha} > dependencies_sha.sh
	imcntk_sha=$$(git ls-remote https://github.com/piloubazin/imcn-imaging.git master | head -n 1 | cut -c1-7) && \
		echo imcntk_sha=$${imcntk_sha} >> dependencies_sha.sh

smoke_tests:
	python3 examples/example_01_tissue_classification.py
	python3 examples/example_02_cortical_depth_estimation.py
	python3 examples/example_03_brain_coregistration.py
	python3 examples/example_04_multiatlas_segmentation.py
	python3 examples/example_05_vascular_reconstruction.py
	python3 examples/example_06_dots_segmentation.py
	python3 examples/example_07_massp_subcortex_parcellation.py 

# =============================================================================
# Docker related content

.PHONY: Dockerfile docker_build clean_tests

clean_tests:
	rm -rf nighres_examples

Dockerfile:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base debian:stretch-slim \
		--pkg-manager apt \
		--install "openjdk-8-jdk git wget build-essential software-properties-common libffi-dev" \
		--miniconda \
			create_env="nighres" \
			conda_install="python=3.9 pip jcc Nilearn" \
			activate="true" \
		--env JAVA_HOME="/docker-java-home" \
		--env JCC_JDK="/docker-java-home" \
		--run 'ln -svT "/usr/lib/jvm/java-8-openjdk-$$(dpkg --print-architecture)" /docker-java-home' \
		--copy build.sh cbstools-lib-files.sh imcntk-lib-files.sh dependencies_sha.sh setup.py setup.cfg MANIFEST.in README.rst LICENSE /nighres/ \
		--copy nighres /nighres/nighres \
		--workdir /nighres \
		--run "conda init && . /root/.bashrc && conda activate nighres && conda info --envs && ./build.sh && rm -rf cbstools-public imcn-imaging nighresjava/build nighresjava/src" \
		--miniconda \
			use_env="nighres" \
			conda_install="jupyterlab" \
			pip_install="." \
		--copy docker/jupyter_notebook_config.py /etc/jupyter \
		--expose 8888 \
		--user neuro \
		--workdir /home/neuro > Dockerfile

docker_build: Dockerfile
	docker build . -t nighres:latest

docker_run: docker_build
	mkdir -p $$PWD/examples
	docker run -it --rm \
		--publish 8888:8888 \
		--volume $$PWD/examples:/home/neuro/examples \
		nighres:latest \
			jupyter-lab --no-browser --ip 0.0.0.0
docker_test: docker_test_cortical_depth_estimation docker_test_coregistration

docker_test_classification: docker_build
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres_examples \
		nighres:latest \
			python3 /examples/example_01_tissue_classification.py 

docker_test_cortical_depth_estimation: docker_test_classification
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres_examples \
		nighres:latest \
			python3 /examples/example_02_cortical_depth_estimation.py 

docker_test_coregistration:
	mkdir -p $$PWD/nighres_examples
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_examples:/home/neuro/nighres_examples \
		nighres:latest \
			python3 /examples/example_03_brain_coregistration.py
