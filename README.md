# A Multimodal fMRI Approach to Spatial Memory Processing: Task Activation and Network Connectivity in ADHD vs. Controls

## Project Overview

This study investigates neural network alterations associated with spatial working memory processing in Attention-Deficit/Hyperactivity Disorder (ADHD) using a comprehensive fMRI analysis approach. The project examines how brain activation patterns and functional connectivity differ between neurotypical controls and individuals with ADHD during rest and spatial memory capacity tasks.

## Research Questions

1. How do spatial working memory networks differ between ADHD and control populations during task performance?
2. What connectivity patterns emerge during rest versus active spatial memory processing?
3. Can machine learning approaches identify reliable neural signatures that distinguish ADHD from neurotypical brain function?



# Analysis Pipeline:

1. First-level analyses characterize individual subject activation patterns contrasting task versus rest conditions. 
2. Second-level group analyses examine main effects and interactions between diagnostic group and task condition. 
3. The machine learning model (CNN-based) learns from the the COPE data generated from first-level analysis to make the diagnosis (ADHD vs Healthy Control)

# Environment setup

The python environment is managed by `uv`, a command-line python package manager. You can follow the  [Offical Instruction](https://docs.astral.sh/uv/getting-started/installation/) to install it

```bash
# Then run the following command to install required packages
uv sync 
```

#  First/Second Level Analysis Result Reproduction (Or EDA, in terms of ML)

For first and second-level analysis, you will need to install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/), since the installation is complicated on different machine and platforms, we will not cover it in this section.

Make sure the necessary tools is available in the `PATH` variable 

```bash
bash prep.sh # Download preprocessed data
```

> We also provided an option to preprocess all raw data locally using fmriprep-docker by setting the environment variable `LOCAL_DERIVATIVES` and then run `prep.py`.

After all the data are downloaded, follow the jupyter notebook 
1. `overview.ipynb` Data overview (check group size and remove invalid data)
2. `1st_level.ipynb`
3. `2nd_level.ipynb`

# ML Results Reproduction

You can train directly without doing first/second level analysis (they served as EDA)

## Demo, Demo, Demo
Since the dataset is over 100GBs, we provide a demo pipeline. You don't need 

```bash
make demo_data # download demo data
make demo # run analysis and generate results
```
## Full training and testing 

```bash
make data
make full
```