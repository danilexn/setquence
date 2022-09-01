#!/bin/bash
# This generates sbatch files specific to alpha

set -e

PYTHONEXEC='/home/dale016c/.conda/envs/dnabert/bin/python'
CONFIG_FILE=''
EXPERIMENT_DIR=''
NODES=1
GPUS_PER_NODE=4
PROJECT=''
PORT=12345

print_usage() {
  printf "Usage: train.sh -c [route to .config file] -e [route to experiment dir] -g [number of GPUS per node] -n [number of nodes] -a [project or account name] -p [port] -s [python executable location]"
}

while getopts 'c:e:g:n:a:p:s:' flag; do
  case "${flag}" in
    c) CONFIG_FILE="${OPTARG}" ;;
    e) EXPERIMENT_DIR="${OPTARG}" ;;
    g) GPUS_PER_NODE="${OPTARG}" ;;
    n) NODES="${OPTARG}" ;;
    a) PROJECT="${OPTARG}" ;;
    p) PORT="${OPTARG}" ;;
    s) PYTHONEXEC="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

CPUS=2
NTASKS=$(( NODES * GPUS_PER_NODE ))
MINCPUS=$(( CPUS * GPUS_PER_NODE ))

jobname=$(basename -- "$CONFIG_FILE")
UIDjob=$(uuidgen)

echo "#!/bin/bash" > .train_jobscript_$UIDjob.sh
echo "#SBATCH --job-name=$jobname" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --partition=alpha" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH -A $PROJECT" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --nodes=$NODES" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --gres=gpu:$GPUS_PER_NODE" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --ntasks=$NTASKS" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --mincpus=$MINCPUS" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --time=07:30:00" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --mem-per-cpu=10G" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --mail-type=all" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH --mail-user=daniel.leon-perinan@mailbox.tu-dresden.de" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH -e slurm-logs/$jobname-%j.err" >> .train_jobscript_$UIDjob.sh
echo "#SBATCH -o slurm-logs/$jobname-%j.out" >> .train_jobscript_$UIDjob.sh

echo "module load modenv/hiera" >> .train_jobscript_$UIDjob.sh
echo "export SLURM_MASTER_PORT=$PORT" >> .train_jobscript_$UIDjob.sh
echo "srun $PYTHONEXEC -m setquence -c $CONFIG_FILE -e $EXPERIMENT_DIR" >> .train_jobscript_$UIDjob.sh

sbatch .train_jobscript_$UIDjob.sh
rm .train_jobscript_$UIDjob.sh
