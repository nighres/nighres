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

.PHONY: Dockerfile NighresSingularity.def docker_build clean_tests

clean_tests:
	rm -rf nighres_examples

Dockerfile:
	bash create_container_recipe.sh docker

NighresSingularity.def:
	bash create_container_recipe.sh singularity

docker_build: Dockerfile
	docker build . -t nighres:latest

NighresSingularity.sif:
	sudo singularity build Nighres.sif NighresSingularity.def

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
