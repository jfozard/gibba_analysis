#!/bin/bash

./make_container.sh containers/meshproject.img Singularity_meshproject
cp containers/meshproject.img $BLADDER_DATA/containers/meshproject.img

./make_container.sh containers/pythonspm.img Singularity_pythonspm
cp containers/pythonspm.img $BLADDER_DATA/containers/pythonspm.img

./make_container.sh containers/surfacespm.img Singularity_surfacespm
cp containers/surfacespm.img $BLADDER_DATA/containers/surfacespm.img

./make_container.sh containers/timagetk.img Singularity_timagetk
cp containers/timagetk.img $BLADDER_DATA/containers/timagetk.img
