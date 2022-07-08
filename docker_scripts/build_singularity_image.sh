#!/bin/bash
docker build -f Dockerfile.base -t drake-tamp-base .
docker run --privileged --rm -w /home/singularity -v `pwd`:/home/singularity -v /var/run/docker.sock:/var/run/docker.sock quay.io/singularity/singularity:v3.8.2-slim build drake-tamp-sing.sif drake-tamp-sing.def
