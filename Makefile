.PHONY: clean install clean_env_file

clean:
	rm -rf nighresjava/
	rm -rf cbstools-public/
	rm -rf imcn-imaging/
	rm -rf fbpa-tools/
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
	python3 examples/testing_01_quantitative_mri.py
	python3 examples/testing_02_cortical_laminar_analysis.py
	python3 examples/testing_03_brain_slab_coregistration.py
	python3 examples/testing_04_massp_subcortex_parcellation.py 

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
docker_test: docker_test_cortical_laminar_analysis docker_test_slab_coregistration

docker_test_quantitative_mri: docker_build
	mkdir -p $$PWD/nighres_testing
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_testing:/home/neuro/nighres_testing \
		nighres:latest \
			python3 /examples/testing_01_quantitative_mri.py 

docker_test_cortical_laminar_analysis: docker_test_quantitative_mri
	mkdir -p $$PWD/nighres_testing
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_testing:/home/neuro/nighres_testing \
		nighres:latest \
			python3 /examples/testing_02_cortical_laminar_analysis.py 

docker_test_slab_coregistration: 
	mkdir -p $$PWD/nighres_testing
	docker run -it --rm \
		--volume $$PWD/examples:/examples \
		--volume $$PWD/nighres_testing:/home/neuro/nighres_testing \
		nighres:latest \
			python3 /examples/testing_03_brain_slab_coregistration.py
