#!/bin/bash

cd "$(dirname $0)"
source ../campaign-env.sh
source group-env.sh

if [ -f "$CODAR_CHEETAH_MACHINE_CONFIG" ]; then
    source "$CODAR_CHEETAH_MACHINE_CONFIG"
fi

# Don't submit job if all experiments have been run
if [ -f codar.workflow.status.json ]; then
    grep state codar.workflow.status.json | grep -q 'not_started'
    if [ $? != 0 ]; then
        echo "No more experiments remaining. Skipping group .."
        exit
    fi
fi

# Copy the env setup to the Sweep Group
if [ -n "$CODAR_CHEETAH_APP_CONFIG" ]; then
  ENV_SETUP_SCRIPT="codar.savanna.env_setup.$CODAR_CHEETAH_MACHINE_NAME"
  cp "$CODAR_CHEETAH_APP_CONFIG" "$ENV_SETUP_SCRIPT"
fi

# convert walltime from seconds to HH:MM:SS format needed by PBS

secs=$CODAR_CHEETAH_GROUP_WALLTIME
PBS_WALLTIME=$(printf '%02d:%02d:%02d\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60)))

OUTPUT=$(qsub \
        -A $CODAR_CHEETAH_SCHEDULER_ACCOUNT \
        -q $CODAR_CHEETAH_SCHEDULER_QUEUE \
        -N "$CODAR_CHEETAH_CAMPAIGN_NAME-$CODAR_CHEETAH_GROUP_NAME" \
        -l nodes=$CODAR_CHEETAH_GROUP_NODES \
        -l walltime=$PBS_WALLTIME \
        run-group.pbs)

rval=$?

if [ $rval != 0 ]; then
    echo "SUBMIT FAILED:"
    echo $OUTPUT
    exit $rval
fi

JOBID=$OUTPUT

echo "PBS:$JOBID" > codar.cheetah.jobid.txt
