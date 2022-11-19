#!/bin/bash
# module swap cluster/slaking # for debug

#PBS -N MIL_NN # job name
#PBS -l nodes=1:ppn=5 # single-node job, try on 5 cores
#PBS -l walltime=0:02:00 # max. 2 min of wall time
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b # also loads numpy

# copy this file, all input data and code from root dir to working dir. Should really keep the folder structure consistent...
mkdir $TMPDIR/src
mkdir $TMPDIR/input
mkdir $TMPDIR/output
cp $VSC_HOME/projects/dummy/src/main.py $TMPDIR/src/main.py
cp $VSC_HOME/projects/dummy/main_job.sh $TMPDIR/src/main_job.sh

# go to temporary working directory (on local disk) & run Python code
cd $TMPDIR
python src/main.py > output/console.txt

#copy all files to data dir
mkdir $VSC_DATA/$PBS_JOBID
mkdir $VSC_DATA/$PBS_JOBID/src
mkdir $VSC_DATA/$PBS_JOBID/input
mkdir $VSC_DATA/$PBS_JOBID/output
cp src/main.py $VSC_DATA/$PBS_JOBID/src/main.py
cp src/main_job.sh $VSC_DATA/$PBS_JOBID/src/main_job.sh
cp output/data.txt $VSC_DATA/$PBS_JOBID/output/data.txt
cp output/console.txt $VSC_DATA/$PBS_JOBID/output/console.txt
mv $PBS_O_WORKDIR/MIL_NN.o$PBS_JOBID $VSC_DATA/$PBS_JOBID/output/MIL_NN.o$PBS_JOBID
mv $PBS_O_WORKDIR/MIL_NN.e$PBS_JOBID $VSC_DATA/$PBS_JOBID/output/MIL_NN.e$PBS_JOBID