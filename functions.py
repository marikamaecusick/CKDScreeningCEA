import pandas as pd
import numpy as np

CYCLE_LENGTH = 1/4.
#ACM_REDUCTION = 1

# def calculate_hr_prob(prob_death, stage, HAZARDS_RATIOS, treated):
    
#     rate_control = -np.log(1-prob_death)
#     rate_treatment = HAZARDS_RATIOS[stage]*rate_control
#     if treated == True: 
#         rate_treatment = rate_treatment*ACM_REDUCTION
#     new_prob = 1 - np.exp(-rate_treatment)
#     return new_prob


#apply reduction of dapa in CKD progression 
def reduction_rate2(prob, value):
    
    if prob == 1:
        return prob
    else:
        newprob = -np.log(1. - prob)*value
    
        newprob = 1- np.exp(-newprob)
        return newprob
    
#convert mortality rates for the cycle length
def convert_to_cycle(prob):
    annual_rate = -np.log(1. - prob)
    monthly_rate = annual_rate*CYCLE_LENGTH 
    return 1 - np.exp(-monthly_rate) 
    
#function to convert the lifetable
def convert_life_cycles(life_table):
    death_cycle= [0 for i in range(len(life_table))]
    for i in range(len(life_table)):
        p1 = life_table['qx'].iloc[i]
        death_cycle[i] = convert_to_cycle(p1)
    life_table['qx_cycle'] = pd.Series(death_cycle, index = life_table.index)
    
    return life_table



