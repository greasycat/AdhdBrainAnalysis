#! /bin/bash

# define a variable called contrast_index
contrast_index=$1
task_dir=$2
task_name=$3

# Find all the tstat$contrast_index$.nii.gz files in the task_dir/*/task_name or its subdirectories
tstat_files=$(find $task_dir/*/$task_name -name "tstat${contrast_index}.nii.gz")

# create a new directory called task_name/contrast_index
mkdir -p inspect/$task_name/$contrast_index
# Copy the tstat files with new names based on their original path
for tstat_file in $tstat_files; do
    # Extract the subfolder name (e.g., sub-001 from task_dir/sub-001/task_name/...)
    subfolder=$(echo $tstat_file | sed "s|$task_dir/||" | cut -d'/' -f1)
    # Create new filename with subfolder prefix
    new_name="${subfolder}_tstat${contrast_index}.nii.gz"
    cp $tstat_file inspect/$task_name/$contrast_index/$new_name
done