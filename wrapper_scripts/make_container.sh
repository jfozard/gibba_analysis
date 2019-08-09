#!/bin/bash
#sudo singularity create -s 4096 $1
#sudo singularity bootstrap $1 $2
sudo singularity build -w $1 $2
