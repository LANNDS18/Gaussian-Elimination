#!/bin/bash -l

# COMP328 High Performance Computing - Programming Assignment
# Bash script to run the OpenMP program
# Author: Wuwei Zhang

# Specific course queue and max wallclock time
#SBATCH -p course -t 2

# Defaults on Barkla (but set to be safe)
## Specify the current working directory as the location for executables/files
#SBATCH -D ./
## Export the current environment to the compute node
#SBATCH --export=ALL

# load modules intel compiler
module load compilers/intel/2019u5

## SLURM terms
## nodes            relates to number of nodes
## ntasks-per-node  relates to MPI processes per node
## cpus-per-task    relates to OpenMP threads (per MPI process)

echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of threads or processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"

# parallel using OpenMP
SRC=$1
EXE=${SRC%%.c}.exe
rm -f ${EXE}
echo compiling $SRC to $EXE
icc -qopenmp -O0 $SRC  -o $EXE
echo
echo ------------------------------------

if test -x $EXE; then
      # set number of threads
      export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
      echo using ${OMP_NUM_THREADS} OpenMP threads
      echo
      echo
      echo Multiple execution..
      echo
      echo
      # run multiple times
      for i in {1..5};
        do
          size=$((200*$i))
          echo "Run $i"
          echo "The size of the matrix is $size"
          ./${EXE} $size;
        done

else
     echo $SRC did not built to $EXE
fi