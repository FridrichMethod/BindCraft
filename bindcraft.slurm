#!/bin/bash
#SBATCH --job-name=bindcraft
#SBATCH --output=bindcraft_%j.out
#SBATCH --error=bindcraft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@stanford.edu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

# Initialise environment and modules
source ~/.bashrc
conda activate ./bindcraft
CONDA_BASE=$(conda info --base)
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# alternatively you can source the environment directly
# source /path/to/mambaforge/bin/activate /path/to/mambaforge/envs/BindCraft

# Get the directory where the bindcraft script is located
SCRIPT_DIR=$(dirname "$0")

# Parsing command line options
SETTINGS=""
FILTERS=""
ADVANCED=""
TEMP=$(getopt -o s:f:a: --long settings:,filters:,advanced: -n 'bindcraft.slurm' -- "$@")
eval set -- "$TEMP"

while true; do
    case "$1" in
        -s | --settings)
            SETTINGS="$2"
            shift 2
            ;;
        -f | --filters)
            FILTERS="$2"
            shift 2
            ;;
        -a | --advanced)
            ADVANCED="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid Option"
            exit 1
            ;;
    esac
done

# Ensure that SETTINGS is not empty
if [ -z "$SETTINGS" ]; then
    echo "Error: The -s or --settings option is required."
    exit 1
fi

echo "Running the BindCraft pipeline"
python -u "${SCRIPT_DIR}/bindcraft.py" --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}"
