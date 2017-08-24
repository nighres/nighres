sudo docker run -ti ubuntu:14.04
sudo apt-get update -qq && apt-get install -y python python-pip python-dev build-essential software-properties-common
sudo add-apt-repository ppa:openjdk-r/ppa && apt-get update -qq && apt-get install -y openjdk-8-jdk
sudo pip install --allow-all-external --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nighres
