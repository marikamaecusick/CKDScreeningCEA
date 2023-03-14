# Cost-effectiveness analysis of population-wide screening for CKD 

## This repository consists of data and code used for our paper "Cost-effectiveness of population-wide screening for chronic kidney disease." Special thanks to my co-authors Rebecca Tisdale, Glenn Chertow, Doug Owens, and Jeremy Goldhaber-Fiebert.

Our code is written in python version 3.9.1. *We plan to update this GitHub with python virtual environment instructions to ensure package and version compatibility.*

Our probabilistic sensitivity analysis (PSA) relies on 10,000 PSA parameter sets. The python code files run 1,000 iterations. To run all 10,000 PSA parameter sets, you will need to alter the starting index appropriately. 

#### For the screening strategies that only rely on ACE inhibitors and ARB therapy: 

python PSA_ace_run_2021_overall.py -d "SIR_parameters.xlsx" -p "PSA.xlsx" -s "[YOUR_SAVENAME].xlsx" -sa "35" -si "0"

- d: SIR parameter data set 
- p: PSA input data set 
- s: excel file to save to: ".xlsx" file 
- sa: model starting age, choose between "35", "45", "55", "65", and "75" 
- si: starting index, choose between "0", "1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000"

#### For the screening strategies that only rely on ACE inhibitors and ARB therapy + SGLT2 inhibitors: 

python PSA_dapa_run_2021_overall.py -d "SIR_parameters.xlsx" -p "PSA.xlsx" -s "[YOUR_SAVENAME].xlsx" -sa "35" -si "0"

- d: SIR parameter data set 
- p: PSA input data set 
- s: excel file to save to: ".xlsx" file 
- sa: model starting age, choose between "35", "45", "55", "65", and "75" 
- si: starting index, choose between "0", "1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000"
