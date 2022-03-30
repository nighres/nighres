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

.PHONY: Dockerfile_dev docker_build

Dockerfile_dev:
	docker run --rm repronim/neurodocker:0.7.0 generate docker \
		--base debian:stretch-slim \
		--pkg-manager apt \
		--install "openjdk-8-jdk git wget build-essential software-properties-common libffi-dev" \
		--copy requirements.txt /tmp \
		--run 'python --version' \
		--run "pip install -r /tmp/requirements.txt" > Dockerfile_dev

docker_dev:
	docker build . -f Dockerfile_dev -t laynii:dev
