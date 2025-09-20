#!/bin/bash

export FSLDIR=/opt/fsl
export PATH=$FSLDIR/bin:$PATH
source $FSLDIR/etc/fslconf/fsl.sh
fsl -V