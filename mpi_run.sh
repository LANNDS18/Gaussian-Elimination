#!/bin/bash -l

# Specific course queue and max wallclock time
#SBATCH -p course -t 2

# Defaults on Barkla (but set to be safe)
## Specify the current working directory as the location for executables/files
#SBATCH -D ./
## Export the current environment to the compute node
#SBATCH --export=ALL

# load modules
## intel compiler
module load compilers/intel
## intel mpi wrapper and run time
module load mpi/intel-mpi/2019u5/bin


## SLURM terms
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



export numMPI=${SLURM_NTASKS:-1} # if '-n' not used then default to 1

echo
echo
mpiicc -O0 $SRC -o $EXE -lm
echo
echo



if test -x $EXE; then
      for i in {1..5};
          do
            size=$((200*$i))
            echo
            echo Process $i
            echo The size of the matrix is $size
            mpirun -np ${numMPI} ./${EXE} $size;
            echo
            echo
      done
else
     echo $SRC did not built to $EXE
fi