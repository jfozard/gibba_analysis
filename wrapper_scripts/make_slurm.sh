#!/bin/bash
scp  Singularity_surfacespm fozardj@v0768.nbi.ac.uk:
scp  ~/.ssh/id_rsa fozardj@v0768.nbi.ac.uk:
ssh v0768.nbi.ac.uk <<EOF
sudo singularity build segment.img Singularity_surfacespm
EOF
scp fozardj@v0768.nbi.ac.uk:segment.img ~/bladder_new/containers
chmod a+x ~/bladder_new/containers/segment.img
