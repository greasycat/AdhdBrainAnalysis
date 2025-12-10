#!/bin/bash
DATA_DIR=data
PREP_DIR=derivatives

# if an export variable LOCAL_DERIVATIVES is set, then generatw the derivatives from the local directory
if [ -n "$LOCAL_DERIVATIVES" ]; then
    PREP_WORKDIR=workdir/derivatives

    mkdir -p $PREP_WORKDIR
    mkdir -p $PREP_DIR

    uv run fmriprep-docker $DATA_DIR $PREP_DIR participant --participant-label $(cat control_subjects_ids.txt) $(cat adhd_subjects_ids.txt) \
        --skip-bids-validation \
        -t rest \
        -w $PREP_WORKDIR \
        --output-spaces T1w MNI152NLin2009cAsym \
        --fs-license-file license.txt

    uv run fmriprep-docker $DATA_DIR $PREP_DIR participant --participant-label $(cat control_subjects_ids.txt) $(cat adhd_subjects_ids.txt) \
        --skip-bids-validation \
        -t rest \
        -w $PREP_WORKDIR \
        --output-spaces T1w MNI152NLin2009cAsym \
        --fs-license-file license.txt
else
    uv run download_derivatives.py --ids control_subjects_ids.txt adhd_subjects_ids.txt \
        --s3-uri s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/ \
        --type fmriprep \
        --tasks rest scap stopsignal taskswitch \
        --output-dir $PREP_DIR 

fi