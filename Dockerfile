# Start from the official Debian image
FROM debian:bullseye

# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    openjdk-17-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Adjust JAVA_HOME if necessary and create a symbolic link to match JCC's expected JDK path
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN ln -s $JAVA_HOME /usr/lib/jvm/temurin-17-jdk-amd64

# Install JCC
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install jcc

# Clone the nighres repository
RUN git clone https://github.com/nighres/nighres

# Change directory into the cloned repository, run the build script, and install nighres
WORKDIR /nighres
RUN ./build.sh && \
    python3 -m pip install .
