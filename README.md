# Cost-effectiveness analysis of population-wide screening for CKD 

## This repository contains the data and code used to evaluate the cost-effectiveness of population-wide screening for chronic kidney disease. Special thanks to my co-authors Rebecca Tisdale, Glenn Chertow, Doug Owens, Joshua Salomon and Jeremy Goldhaber-Fiebert.

### Our analysis entitled "[Cost-effectiveness of population-wide screening for chronic kidney disease](https://www.acpjournals.org/doi/abs/10.7326/m22-3228)" was published in the Annals of Internal Medicine in 2023. 

### Since then, we updated our analysis to evaluate the optimal screening initiation age. Our working paper is titled "When to start population-wide screening for CKD: a cost-effectiveness analysis." 

Our code is written in python version 3.9.1. *We plan to update this GitHub with python virtual environment instructions to ensure package and version compatibility.*

Our probabilistic sensitivity analysis (PSA) relies on 10,000 PSA parameter sets. The python code files run 1,000 iterations. To run all 10,000 PSA parameter sets, you will need to alter the starting index appropriately. 

#### For the screening strategies that only rely on ACE inhibitors and ARB therapy: 

python PSA_ace_run_overall.py -d "SIR_parameters.xlsx" -p "PSA.xlsx" -s "[YOUR_SAVENAME].xlsx" -sa "35" -si "0"

- d: SIR parameter data set 
- p: PSA input data set 
- s: excel file to save to: ".xlsx" file 
- sa: model starting age, choose between "35", "45", "55", "65", and "75" 
- si: starting index, choose between "0", "1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000" (The file will run 1,000 iterations at a time)

#### For the screening strategies that only rely on ACE inhibitors and ARB therapy + SGLT2 inhibitors: 

python PSA_dapa_run_overall.py -d "SIR_parameters.xlsx" -p "PSA.xlsx" -s "[YOUR_SAVENAME].xlsx" -sa "35" -si "0"

- d: SIR parameter data set 
- p: PSA input data set 
- s: excel file to save to: ".xlsx" file 
- sa: model starting age, choose between "35", "45", "55", "65", and "75" 
- si: starting index, choose between "0", "1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000" (The file will run 1,000 iterations at a time)
