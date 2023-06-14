#! /bin/bash

export RESULT_DIR="/data/results/event_feature_tracking" && \
export DATASET="shapes_6dof" && \
export METHOD="haste_correlation" && \
matlab -nodisplay -nosplash -nodesktop -nojvm \
 -r "eval_haste($RESULT_DIR, $DATASET, $METHOD, true); catch; end; quit"
