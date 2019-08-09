#!/bin/bash
scp  Singularity_meshproject fozardj@v0768.nbi.ac.uk:
scp  ~/.ssh/id_rsa fozardj@v0768.nbi.ac.uk:
ssh v0768.nbi.ac.uk <<EOF
sudo singularity build meshproject.img Singularity_meshproject
EOF
scp fozardj@v0768.nbi.ac.uk:meshproject.img ~/bladder_new/containers
chmod a+x ~/bladder_new/containers/meshproject.img
