#!/bin/bash

# Usage:
# ./dependency_launcher.sh -n 5 <file.sub>
# e.g: ./dependency_launcher.sh -n 5 file.sub
# it will call each file with one argument that's corresponding to the index of the job,
# e.g. "file.sub 1" -> "file.sub 2" -> "file.sub 3" -> ...

# Grab command line options
# n: Number of times to submit the job
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

# Shift the command line arguments to get the SUBFILE
shift $((OPTIND-1))
SUBFILE=$1
if [[ -z $SUBFILE ]]; then
  echo "Usage: $(basename "$0") [flags] [sub file] [arguments for sub file]"
  exit 1
fi

# Remove the SUBFILE from the argument list
shift

echo "Calling [$SUBFILE] $N_CALLS times"

# Repeat calls
PREV_JOBID=""
for (( i = 1; i <= $N_CALLS; i++ ))
do
  RSEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
  if [ -z $PREV_JOBID ]; then
    echo "Submitting job ${i}"
    OUTPUT=$(sbatch $SUBFILE "$RSEED")
  else
    echo "Submitting job ${i} w/ dependency on jobid ${PREV_JOBID}"
    OUTPUT=$(sbatch --dependency=afterany:${PREV_JOBID} $SUBFILE "$RSEED")
  fi
  PREV_JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
done

squeue --start -u $(whoami) -l

