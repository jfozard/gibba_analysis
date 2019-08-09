#!/bin/bash
signal=$1
shift
other="$@"
echo $other
ssh slurm <<HERE
sbatch -p rg-sv  <<EOF 
#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1             # How many tasks on each node
#SBATCH --time=100:00:00
#SBATCH --mem 15000
echo $other
sleep 20
time $other
touch $signal
EOF
HERE
