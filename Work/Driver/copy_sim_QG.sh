#!/bin/sh
set -e

RUNID_START=10  # Start value for runid (trained model num)
RUNID_END=10  # End value for runid

BASE_DIR="/home/mjmvega/Codes/QG/Work/"  # Where to copy code from
RESULTS_ROOT="/gdata/Results/mjmvega/QG/" #Results directory

for (( runid=${RUNID_START}; runid <= ${RUNID_END}; runid++ )); do
    padded_r=$(printf "%05d" "$runid") 
    RESULTS_DIR="${RESULTS_ROOT}/Run${padded_r}"
    RUN_DIR="${RESULTS_DIR}/Code/"
    
    mkdir -p "${RUN_DIR}" "${RESULTS_ROOT}/Logs"
    
    echo "Copying code to ${RUN_DIR} ..."
    rsync -a --delete --exclude '__pycache__' --exclude '.git' --exclude '.ipynb_checkpoints' "${BASE_DIR}/" "${RUN_DIR}/"
    
    padded_r=$(printf "%04d" "$runid") #Better for qstat display
    
    qsub -N "SimQG${padded_r}" -v RUN_DIR="${RUN_DIR}" "SimQG" "${runid}"
    
    sleep 3
done
