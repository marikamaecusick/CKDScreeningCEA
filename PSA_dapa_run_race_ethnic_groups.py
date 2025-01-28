import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import time

from argparse import ArgumentParser

import functions
from functions import *


def calculate_hr_prob(prob_death, stage, HAZARDS_RATIOS, treated, ACM_REDUCTION):
    rate_control = -np.log(1 - prob_death)
    rate_treatment = HAZARDS_RATIOS[stage] * rate_control
    if treated == True:
        rate_treatment = rate_treatment * ACM_REDUCTION
    new_prob = 1 - np.exp(-rate_treatment)
    return new_prob


def create_transition_matrix(a, MALE):
    transition_matrix = np.zeros((len(HEALTH_STATES), len(HEALTH_STATES)))

    if MALE == True:
        p_dead = male_life_table[male_life_table["Age"] == AGE_LIST[a]][
            "qx_cycle"
        ].iloc[0]
    else:
        p_dead = female_life_table[female_life_table["Age"] == AGE_LIST[a]][
            "qx_cycle"
        ].iloc[0]

    if a <= 39:
        eGFR_REDUCTION_DAPA = eGFR_REDUCTION_DAPA_dict[39]
        ACM_REDUCTION = ACM_REDUCTION_dict[39]
    elif a <= 49:
        eGFR_REDUCTION_DAPA = eGFR_REDUCTION_DAPA_dict[49]
        ACM_REDUCTION = ACM_REDUCTION_dict[49]
    elif a <= 59:
        eGFR_REDUCTION_DAPA = eGFR_REDUCTION_DAPA_dict[59]
        ACM_REDUCTION = ACM_REDUCTION_dict[59]
    elif a <= 69:
        eGFR_REDUCTION_DAPA = eGFR_REDUCTION_DAPA_dict[69]
        ACM_REDUCTION = ACM_REDUCTION_dict[69]
    else:
        eGFR_REDUCTION_DAPA = eGFR_REDUCTION_DAPA_dict[70]
        ACM_REDUCTION = ACM_REDUCTION_dict[70]

    for i in range(len(HEALTH_STATES)):
        # if we are not dead and we are not in kidney failure on KRT
        if i != DEAD_INDEX and i != 48 and i != 49 and i != 50:
            p_alb = ALBUMIN_PROBS[HEALTH_STATES[i]]
            p_egfr = eGFR_PROBS[HEALTH_STATES[i]]

            if i in stage_1_age_acceleration_list:
                if a < YOUNG_AGE:
                    p_egfr = eGFR_PROBS[HEALTH_STATES[i]]
                elif a >= YOUNG_AGE and a <= OLD_AGE:
                    p_egfr = STAGE_1_ACCELERATION_LINEAR(a)
                    if p_egfr < 0:
                        print("WRONG")
                else:
                    p_egfr = eGFR_PROBS["STAGE_1_OLDER"]

            if i in stage_2_age_acceleration_list:
                if a < YOUNG_AGE:
                    p_egfr = eGFR_PROBS[HEALTH_STATES[i]]
                elif a >= YOUNG_AGE and a <= OLD_AGE:
                    p_egfr = STAGE_2_ACCELERATION_LINEAR(a)
                    if p_egfr < 0:
                        print("WRONG")
                else:
                    p_egfr = eGFR_PROBS["STAGE_2_OLDER"]

            if i in stage_3_age_acceleration_list:
                if a < YOUNG_AGE:
                    p_egfr = eGFR_PROBS[HEALTH_STATES[i]]
                elif a >= YOUNG_AGE and a <= OLD_AGE:
                    p_egfr = STAGE_3_ACCELERATION_LINEAR(a)
                    if p_egfr < 0:
                        print("WRONG")
                else:
                    p_egfr = eGFR_PROBS["STAGE_3_OLDER"]

            ##according to the albuminuria stage, we are going to progress faster/slower in CKD
        if "micro" in HEALTH_STATES[i]:
            p_egfr = reduction_rate2(p_egfr, ALBUMIN_ACCELERATION[HEALTH_STATES[i]])

        if "macro" in HEALTH_STATES[i]:
            p_alb = 0
            p_egfr = reduction_rate2(p_egfr, ALBUMIN_ACCELERATION[HEALTH_STATES[i]])

        if (
            HEALTH_STATES[i] in detected_treated_accelerated
            and a == TREATMENT_ACCELERATION_AGE
        ):
            IMMEDIATE_TREATMENT[HEALTH_STATES[i]] = reduction_rate2(
                IMMEDIATE_TREATMENT[HEALTH_STATES[i]], TREATMENT_ACCELERATION_HR
            )

        if (
            HEALTH_STATES[i] in detected_treated_accelerated
            and a == DETECTION_ACCELERATION_AGE
        ):
            DETECTION_ACCELERATION[HEALTH_STATES[i]] = reduction_rate2(
                DETECTION_ACCELERATION[HEALTH_STATES[i]], DETECTION_ACCELERATION_HR
            )

        if HEALTH_STATES[i] in STAGES_WITH_DAPA:
            p_disc = PROB_DISCONTINUE2
        else:
            p_disc = PROB_DISCONTINUE

        # CKD Stage 1-3b
        if i < 12:
            # detected
            transition_matrix[i][i + 15] = DETECTION_ACCELERATION[HEALTH_STATES[i]] * (
                1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]]
            )
            # detected and treated
            transition_matrix[i][i + 30] = (
                DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            # albumin_transitions
            transition_matrix[i][i + 1] = p_alb * (
                1 - DETECTION_ACCELERATION[HEALTH_STATES[i]]
            )
            transition_matrix[i][i + 16] = (
                p_alb
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )
            transition_matrix[i][i + 31] = (
                p_alb
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            # eGFR transitions
            transition_matrix[i][i + 3] = p_egfr * (
                1 - DETECTION_ACCELERATION[HEALTH_STATES[i]]
            )

            transition_matrix[i][i + 18] = (
                p_egfr
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )
            transition_matrix[i][i + 33] = (
                p_egfr
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            # both transitions

            transition_matrix[i][i + 4] = (
                p_alb * p_egfr * (1 - DETECTION_ACCELERATION[HEALTH_STATES[i]])
            )
            transition_matrix[i][i + 19] = (
                p_alb
                * p_egfr
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )
            transition_matrix[i][i + 34] = (
                p_alb
                * p_egfr
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )  # p_d

        # Stage 4 (not detected)
        # detected
        if i == 12 or i == 13 or i == 14:
            transition_matrix[i][i + 15] = DETECTION_ACCELERATION[HEALTH_STATES[i]] * (
                1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]]
            )
            # detected and treated
            transition_matrix[i][i + 30] = (
                DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            # albumin_transitions
            transition_matrix[i][i + 1] = p_alb * (
                1 - DETECTION_ACCELERATION[HEALTH_STATES[i]]
            )
            transition_matrix[i][i + 16] = (
                p_alb
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (1 - IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )
            transition_matrix[i][i + 31] = (
                p_alb
                * DETECTION_ACCELERATION[HEALTH_STATES[i]]
                * (IMMEDIATE_TREATMENT[HEALTH_STATES[i]])
            )

            transition_matrix[i][i + 33] = p_egfr

            transition_matrix[i][i + 34] = p_egfr * p_alb

            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )

        # Stage (1-3b - detected)
        if i > 14 and i < 27:
            # detected and treated (no transition)
            transition_matrix[i][i + 15] = TREATED_PROB

            transition_matrix[i][i + 1] = p_alb * (1 - TREATED_PROB)
            transition_matrix[i][i + 16] = p_alb * (TREATED_PROB)

            transition_matrix[i][i + 3] = p_egfr * (1 - TREATED_PROB)
            transition_matrix[i][i + 18] = p_egfr * (TREATED_PROB)

            transition_matrix[i][i + 4] = p_alb * p_egfr * (1 - TREATED_PROB)
            transition_matrix[i][i + 19] = p_alb * p_egfr * (TREATED_PROB)

            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )

        # Stage (4 - detected)
        if i == 27 or i == 28 or i == 29:
            transition_matrix[i][i + 15] = TREATED_PROB

            transition_matrix[i][i + 1] = p_alb * (1 - TREATED_PROB)
            transition_matrix[i][i + 16] = p_alb * (TREATED_PROB)

            transition_matrix[i][i + 18] = p_egfr  

            transition_matrix[i][i + 19] = p_alb * p_egfr  

            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )

        ##Stage 1- 3b detected and treated
        if i > 29 and i < 42:
            transition_matrix[i][i - 15] = p_disc  # PROB_DISCONTINUE

            transition_matrix[i][i + 1] = p_alb * (
                1 - p_disc
            )  # p_alb*(1-PROB_DISCONTINUE)
            transition_matrix[i][i - 14] = p_alb * p_disc  

            if HEALTH_STATES[i] not in NOT_TRUE_CKD_COLS_TREATED:
                if (
                    "micro" in HEALTH_STATES[i] or "macro" in HEALTH_STATES[i]
                ) and "Stage_1" not in HEALTH_STATES[i]:
                    transition_matrix[i][i + 3] = reduction_rate2(
                        p_egfr, eGFR_REDUCTION_DAPA
                    ) * (1 - p_disc)
                    
                    transition_matrix[i][i - 12] = reduction_rate2(
                        p_egfr, eGFR_REDUCTION_DAPA
                    ) * (p_disc)
                    

                    transition_matrix[i][i + 4] = (
                        p_alb
                        * reduction_rate2(p_egfr, eGFR_REDUCTION_DAPA)
                        * (1 - p_disc)
                    )
            
                    transition_matrix[i][i - 11] = (
                        p_alb * reduction_rate2(p_egfr, eGFR_REDUCTION_DAPA) * (p_disc)
                    )
             

                    transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                        p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, True, ACM_REDUCTION
                    )

                else:
                    transition_matrix[i][i + 3] = reduction_rate2(
                        p_egfr, eGFR_REDUCTION_ACE
                    ) * (1 - p_disc)
                   
                    transition_matrix[i][i - 12] = reduction_rate2(
                        p_egfr, eGFR_REDUCTION_ACE
                    ) * (p_disc)
                   

                    transition_matrix[i][i + 4] = (
                        p_alb
                        * reduction_rate2(p_egfr, eGFR_REDUCTION_ACE)
                        * (1 - p_disc)
                    )
          
                    transition_matrix[i][i - 11] = (
                        p_alb * reduction_rate2(p_egfr, eGFR_REDUCTION_ACE) * (p_disc)
                    )
                 

                    transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                        p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
                    )

            else:
                transition_matrix[i][i + 3] = p_egfr * (1 - p_disc)
            
                transition_matrix[i][i - 12] = p_egfr * p_disc
     
                transition_matrix[i][i + 4] = p_alb * p_egfr * (1 - p_disc)
               
                transition_matrix[i][i - 11] = p_alb * p_egfr * (p_disc)
              

                transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                    p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
                )

        if i == 42 or i == 43 or i == 44:
            transition_matrix[i][i - 15] = p_disc  # PROB_DISCONTINUE

            transition_matrix[i][i + 1] = p_alb * (
                1 - p_disc
            )  
            transition_matrix[i][i - 14] = p_alb * (p_disc)  

            if "micro" in HEALTH_STATES[i] or "macro" in HEALTH_STATES[i]:
                transition_matrix[i][i + 3] = reduction_rate2(
                    p_egfr, eGFR_REDUCTION_DAPA
                )
                transition_matrix[i][i + 4] = p_alb * reduction_rate2(
                    p_egfr, eGFR_REDUCTION_DAPA
                )
                transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                    p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, True, ACM_REDUCTION
                )
            else:
                transition_matrix[i][i + 3] = reduction_rate2(
                    p_egfr, eGFR_REDUCTION_ACE
                )
                transition_matrix[i][i + 4] = p_alb * reduction_rate2(
                    p_egfr, eGFR_REDUCTION_ACE
                )
                transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                    p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
                )

        # KF PRE-KRT ()
        if i == 45 or i == 46 or i == 47:
            transition_matrix[i][i + 1] = p_alb

            if "micro" in HEALTH_STATES[i] or "macro" in HEALTH_STATES[i]:
                transition_matrix[i][i + 3] = reduction_rate2(
                    p_egfr, eGFR_REDUCTION_DAPA
                )
                transition_matrix[i][i + 4] = p_alb * reduction_rate2(
                    p_egfr, eGFR_REDUCTION_DAPA
                )
                transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                    p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, True, ACM_REDUCTION
                )
            else:
                transition_matrix[i][i + 3] = reduction_rate2(
                    p_egfr, eGFR_REDUCTION_ACE
                )
                transition_matrix[i][i + 4] = p_alb * reduction_rate2(
                    p_egfr, eGFR_REDUCTION_ACE
                )
                transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                    p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
                )

        # KF on KRT (no albumin, micro)
        if i == 48 or i == 49:
            transition_matrix[i][i + 1] = ALBUMIN_PROBS[HEALTH_STATES[i]]
            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )

        # KF on KRT (macro)
        if i == 50:
            transition_matrix[i][DEAD_INDEX] = calculate_hr_prob(
                p_dead, HEALTH_STATES[i], HAZARDS_RATIOS, False, ACM_REDUCTION
            )

        transition_matrix[i][i] = 1 - sum(transition_matrix[i])

    return transition_matrix


def run_model(male, screening, calibration_targets):
    trace = []
    detected_count = [0 for i in range(NUM_ITERATIONS + 1)]
    abnormal_detected_count = [0 for i in range(NUM_ITERATIONS + 1)]
    kf_incidence = [0 for i in range(NUM_ITERATIONS + 1)]
    treated_count = [0 for i in range(NUM_ITERATIONS + 1)]
    misdetected_count = [0 for i in range(NUM_ITERATIONS + 1)]
    mistreated_count = [0 for i in range(NUM_ITERATIONS + 1)]
    total_screened_count = [0 for i in range(NUM_ITERATIONS + 1)]

    this_starting_distribution = np.array(calibration_targets)
    transition_matrix = create_transition_matrix(STARTING_AGE, male)

    newly_detected = 0
    newly_treated = 0
    mistreated = 0
    misdetected = 0
    false_positive_prop = 0
    abnormal_ckd_count = 0
    total_screened = 0
    if screening == True and 0 in SCREENING_AFTER_ITERATIONS:
        (
            this_starting_distribution,
            newly_detected,
            newly_treated,
            misdetected,
            mistreated,
            false_positive_prop,
            abnormal_ckd_count,
            total_screened,
        ) = apply_screening(this_starting_distribution)

    trace.append(this_starting_distribution)

    detected_count[0] = newly_detected
    treated_count[0] = newly_treated
    kf_incidence[0] = 0  
    misdetected_count[0] = false_positive_prop  
    mistreated_count[0] = mistreated
    abnormal_detected_count[0] = abnormal_ckd_count
    total_screened_count[0] = total_screened

    count = 0
    for i in range(NUM_ITERATIONS):
        newly_detected = 0
        newly_treated = 0
        mistreated = 0
        misdetected = 0
        false_positive_prop = 0
        abormal_ckd_count = 0
        total_screened = 0
        if i == 0:
            new_distribution = np.matmul(this_starting_distribution, transition_matrix)
            kf_count = 0
            detected_trk = 0
            treated_trk = 0
            for j in kf_number_cols_list:
                x = transition_matrix[j][48:51]
                kf_count = kf_count + sum(x * this_starting_distribution[j])
            for d in not_detected_cols_list:
                x = transition_matrix[d][detected_or_treated_cols_list]
                x2 = transition_matrix[d][treated_cols_list]
                detected_trk = detected_trk + sum(x * this_starting_distribution[d])
                treated_trk = treated_trk + sum(x2 * this_starting_distribution[d])

        else:
            transition_matrix = create_transition_matrix(
                int(STARTING_AGE + i * CYCLE_LENGTH), male
            )
            new_distribution1 = np.matmul(new_distribution, transition_matrix)
            kf_count = 0
            detected_trk = 0
            treated_trk = 0
            for j in kf_number_cols_list:
                x = transition_matrix[j][48:51]
                kf_count = kf_count + sum(x * new_distribution[j])

            for d in not_detected_cols_list:
                x = transition_matrix[d][detected_or_treated_cols_list]
                x2 = transition_matrix[d][treated_cols_list]
                detected_trk = detected_trk + sum(x * new_distribution[d])
                treated_trk = treated_trk + sum(x2 * new_distribution[d])


            if screening == True and i in SCREENING_AFTER_ITERATIONS:
                (
                    new_distribution1,
                    newly_detected,
                    newly_treated,
                    misdetected,
                    mistreated,
                    false_positive_prop,
                    abnormal_ckd_count,
                    total_screened,
                ) = apply_screening(new_distribution1)

            new_distribution = new_distribution1

        detected_count[i + 1] = newly_detected + detected_trk
        treated_count[i + 1] = newly_treated + treated_trk
        kf_incidence[i + 1] = kf_count
        misdetected_count[i + 1] = false_positive_prop 
        mistreated_count[i + 1] = mistreated
        abnormal_detected_count[i + 1] = abnormal_ckd_count
        total_screened_count[i + 1] = total_screened

        trace.append(new_distribution)

    trace = pd.DataFrame(trace)
    trace.columns = HEALTH_STATES
    trace["kf_incidence"] = kf_incidence
    trace["detected_percent"] = detected_count
    trace["treated_percent"] = treated_count
    trace["misdetected_percent"] = misdetected_count
    trace["mistreated_percent"] = mistreated_count
    trace["abnormal_detected"] = abnormal_detected_count
    trace["total_screened"] = total_screened_count

    total_incidence = [0 for i in range(len(trace))]
    for i in range(len(trace)):
        if i == 0:
            total_incidence[i] = trace["kf_incidence"].iloc[i]
        else:
            total_incidence[i] = total_incidence[i - 1] + trace["kf_incidence"].iloc[i]

    trace["kf_total_incidence"] = total_incidence

    return trace


def apply_screening(distribution):
    correctly_detected_count = 0
    correctly_treated_count = 0
    incorrect_detected_count = 0
    incorrect_treated_count = 0
    abnormal_detected_count = 0

    total_not_detected = 0
    total_screened = 0

    for j in range(len(HEALTH_STATES)):
        # not detected and we detect some people
        if (
            "notdetected" in HEALTH_STATES[j]
            and HEALTH_STATES[j] not in NOT_TRUE_CKD_COLS2
        ):

            this_detection_stage = (
                HEALTH_STATES[j].split("_notdetected")[0] + "_detected"
            )
            this_detection_index = HEALTH_STATES.index(this_detection_stage)

            this_treatment_stage = (
                HEALTH_STATES[j].split("_notdetected")[0] + "_detected_treated"
            )
            this_treatment_index = HEALTH_STATES.index(this_treatment_stage)

            newly_detected_treated = (
                distribution[j] * SCREENING_SENSITIVITY * TREATMENT_INITATION
            )
            newly_detected_untreated = (
                distribution[j] * SCREENING_SENSITIVITY * (1 - TREATMENT_INITATION)
            )

            total_screened = total_screened + distribution[j]

            distribution[this_detection_index] = (
                distribution[this_detection_index] + newly_detected_untreated
            )
            distribution[this_treatment_index] = (
                distribution[this_treatment_index] + newly_detected_treated
            )

            distribution[j] = distribution[j] * (1 - SCREENING_SENSITIVITY)

            correctly_detected_count = (
                correctly_detected_count
                + newly_detected_treated
                + newly_detected_untreated
            )
            correctly_treated_count = correctly_treated_count + newly_detected_treated

            if HEALTH_STATES[j] in ABNORMAL_CKD_COLS:
                abnormal_detected_count = (
                    abnormal_detected_count
                    + newly_detected_treated
                    + newly_detected_untreated
                )

        if HEALTH_STATES[j] in NOT_TRUE_CKD_COLS2:
            total_not_detected = total_not_detected + distribution[j]

            this_detection_stage = (
                HEALTH_STATES[j].split("_notdetected")[0] + "_detected"
            )
            this_detection_index = HEALTH_STATES.index(this_detection_stage)

            this_treatment_stage = (
                HEALTH_STATES[j].split("_notdetected")[0] + "_detected_treated"
            )
            this_treatment_index = HEALTH_STATES.index(this_treatment_stage)

            newly_detected_treated = (
                distribution[j] * (1 - SCREENING_SPECIFICITY) * TREATMENT_INITATION
            )
            newly_detected_untreated = (
                distribution[j]
                * (1 - SCREENING_SPECIFICITY)
                * (1 - TREATMENT_INITATION)
            )

            total_screened = total_screened + distribution[j]

            distribution[this_detection_index] = (
                distribution[this_detection_index] + newly_detected_untreated
            )
            distribution[this_treatment_index] = (
                distribution[this_treatment_index] + newly_detected_treated
            )

            distribution[j] = (
                distribution[j] - newly_detected_treated - newly_detected_untreated
            )

            incorrect_treated_count = incorrect_treated_count + newly_detected_treated
            incorrect_detected_count = (
                incorrect_detected_count
                + newly_detected_treated
                + newly_detected_untreated
            )

    return (
        distribution,
        correctly_detected_count + incorrect_detected_count,
        correctly_treated_count + incorrect_treated_count,
        incorrect_detected_count,
        incorrect_treated_count,
        incorrect_detected_count / total_not_detected,
        abnormal_detected_count,
        total_screened,
    )


def transform_distributions(trace, do_male, screening):
    LY = [0 for i in range(len(trace))]
    CUM_LY = [0 for i in range(len(trace))]
    dis_LY = [0 for i in range(len(trace))]
    dis_CUM_LY = [0 for i in range(len(trace))]
    QALY = [0 for i in range(len(trace))]
    CUM_QALY = [0 for i in range(len(trace))]
    dis_QALY = [0 for i in range(len(trace))]
    dis_CUM_QALY = [0 for i in range(len(trace))]
    COSTS = [0 for i in range(len(trace))]
    CUM_COSTS = [0 for i in range(len(trace))]
    dis_COSTS = [0 for i in range(len(trace))]
    dis_CUM_COSTS = [0 for i in range(len(trace))]

    cum_DAPA_COSTS = [0 for i in range(len(trace))]
    dis_cum_DAPA_COSTS = [0 for i in range(len(trace))]
    screening_COSTS_TRACKER = [0 for i in range(len(trace))]
    dis_screening_COSTS_TRACKER = [0 for i in range(len(trace))]

    dis_LY_detected = [0 for i in range(len(trace))]
    dis_QALY_detected = [0 for i in range(len(trace))]
    dis_cost_detected = [0 for i in range(len(trace))]

    cum_ACE_COSTS = [0 for i in range(len(trace))]
    dis_cum_ACE_COSTS = [0 for i in range(len(trace))]
    diagnosis_COSTS_TRACKER = [0 for i in range(len(trace))]
    dis_diagnosis_COSTS_TRACKER = [0 for i in range(len(trace))]

    for i in range(len(trace)):
        current_age = int(STARTING_AGE + i * CYCLE_LENGTH)

        if do_male == True:
            this_age_qaly = male_life_table[
                male_life_table["Age"] == AGE_LIST[current_age]
            ]["age_qaly"].iloc[0]
            baseline_costs = (
                MEPS_data[MEPS_data["Age"] == current_age]["Male-2021"].iloc[0]
                * CYCLE_LENGTH
                * BASELINE_MULTIPLIER
            )
        else:
            this_age_qaly = female_life_table[
                female_life_table["Age"] == AGE_LIST[current_age]
            ]["age_qaly"].iloc[0]
            baseline_costs = (
                MEPS_data[MEPS_data["Age"] == current_age]["Female-2021"].iloc[0]
                * CYCLE_LENGTH
                * BASELINE_MULTIPLIER
            )

        this_cycle_QALY = 0
        this_cycle_COST = 0

        this_cycle_QALY_detected = 0
        this_cycle_cost_detected = 0
        for j in HEALTH_STATES:
            if j != "Dead":
                this_cycle_QALY = this_cycle_QALY + (
                    STAGE_QALYS[j] * trace[j].iloc[i] * this_age_qaly * CYCLE_LENGTH
                )

                this_cycle_COST = (
                    this_cycle_COST
                    + STAGE_COSTS[j] * trace[j].iloc[i]
                    + baseline_costs * trace[j].iloc[i]
                )

                if j in STAGES_WITH_ACE:
                    this_cycle_COST = this_cycle_COST + ACE_COSTS * trace[j].iloc[i]
                    cum_ACE_COSTS[i] = cum_ACE_COSTS[i] + ACE_COSTS * trace[j].iloc[i]

                if j in STAGES_WITH_DAPA:
                    this_cycle_COST = this_cycle_COST + DAPA_COST * trace[j].iloc[i]

                    # AMONG THOSE ON SGLT2 inhibitors, they face adverse events
                    this_cycle_COST = (
                        this_cycle_COST
                        + trace[j].iloc[i] * UTI_EVENT_PROP * UTI_EVENT_COST
                    )
                    this_cycle_QALY = (
                        this_cycle_QALY
                        + trace[j].iloc[i] * UTI_EVENT_PROP * UTI_EVENT_QALY
                    )

                    this_cycle_COST = (
                        this_cycle_COST
                        + trace[j].iloc[i] * DKA_EVENT_PROP * DKA_EVENT_COST
                    )
                    this_cycle_QALY = (
                        this_cycle_QALY
                        + trace[j].iloc[i] * DKA_EVENT_PROP * DKA_EVENT_QALY
                    )

                    cum_DAPA_COSTS[i] = cum_DAPA_COSTS[i] + DAPA_COST * trace[j].iloc[i]

                if j in detected_or_treated_list:
                    this_cycle_QALY_detected = this_cycle_QALY_detected + (
                        STAGE_QALYS[j] * trace[j].iloc[i] * this_age_qaly * CYCLE_LENGTH
                    )
                    this_cycle_cost_detected = (
                        this_cycle_cost_detected
                        + STAGE_COSTS[j] * trace[j].iloc[i]
                        + baseline_costs * trace[j].iloc[i]
                    )
                    if j in STAGES_WITH_ACE:
                        this_cycle_cost_detected = (
                            this_cycle_cost_detected + ACE_COSTS * trace[j].iloc[i]
                        )
                    if j in STAGES_WITH_DAPA:
                        this_cycle_cost_detected = (
                            this_cycle_cost_detected + DAPA_COST * trace[j].iloc[i]
                        )


        if screening == True and i in SCREENING_AFTER_ITERATIONS_C:
            this_cycle_COST = (
                this_cycle_COST + SCREENING_COST * trace["total_screened"].iloc[i]
            )
            this_cycle_COST = (
                this_cycle_COST + OFFICE_VISIT * trace["total_screened"].iloc[i]
            )

            screening_COSTS_TRACKER[i] = (
                screening_COSTS_TRACKER[i]
                + SCREENING_COST * trace["total_screened"].iloc[i]
                + OFFICE_VISIT * trace["total_screened"].iloc[i]
            )

        if (
            trace["detected_percent"].iloc[i] != 0
            and screening == True
            and i in SCREENING_AFTER_ITERATIONS_C
        ):
            this_cycle_COST = (
                this_cycle_COST + trace["detected_percent"].iloc[i] * SERUM_CR
            )
            this_cycle_COST = (
                this_cycle_COST
                + trace["detected_percent"].iloc[i] * ADVERSE_PERCENTAGE * ADVERSE_COST
            )
            this_cycle_QALY = (
                this_cycle_QALY
                + trace["detected_percent"].iloc[i] * ADVERSE_PERCENTAGE * ADVERSE_QALY
            )

            this_cycle_cost_detected = (
                this_cycle_cost_detected + trace["detected_percent"].iloc[i] * SERUM_CR
            )
            this_cycle_cost_detected = (
                this_cycle_cost_detected
                + trace["detected_percent"].iloc[i] * ADVERSE_PERCENTAGE * ADVERSE_COST
            )

            diagnosis_COSTS_TRACKER[i] = (
                diagnosis_COSTS_TRACKER[i]
                + trace["detected_percent"].iloc[i] * SERUM_CR
            )

            ##the people who do not continue with the drug in the first 3 months incur the costs in the first three months
            this_cycle_COST = this_cycle_COST + (
                trace["detected_percent"].iloc[i] - trace["treated_percent"].iloc[i]
            ) * (ACE_COSTS + DAPA_COST)

            this_cycle_cost_detected = this_cycle_cost_detected + (
                trace["detected_percent"].iloc[i] - trace["treated_percent"].iloc[i]
            ) * (ACE_COSTS + DAPA_COST)

        if (
            trace["abnormal_detected"].iloc[i] != 0
            and screening == True
            and i in SCREENING_AFTER_ITERATIONS_C
        ):
            this_cycle_COST = (
                this_cycle_COST + trace["abnormal_detected"].iloc[i] * ULTRASOUND
            )

            this_cycle_cost_detected = (
                this_cycle_cost_detected
                + trace["abnormal_detected"].iloc[i] * ULTRASOUND
            )

            diagnosis_COSTS_TRACKER[i] = (
                diagnosis_COSTS_TRACKER[i]
                + trace["detected_percent"].iloc[i] * ULTRASOUND
            )

        if i == 0 or i == NUM_ITERATIONS:
            LY[i] = (sum(trace[alive_states].iloc[i]) * CYCLE_LENGTH) / 2
            QALY[i] = this_cycle_QALY / 2
            COSTS[i] = this_cycle_COST / 2

        else:
            LY[i] = sum(trace[alive_states].iloc[i]) * CYCLE_LENGTH
            QALY[i] = this_cycle_QALY
            COSTS[i] = this_cycle_COST

        if i == 0:
            CUM_LY[i] = LY[i]
            dis_LY[i] = LY[i]
            dis_CUM_LY[i] = LY[i]

            CUM_QALY[i] = QALY[i]
            dis_QALY[i] = QALY[i]
            dis_CUM_QALY[i] = QALY[i]

            CUM_COSTS[i] = COSTS[i]
            dis_COSTS[i] = COSTS[i]
            dis_CUM_COSTS[i] = COSTS[i]

            dis_cum_DAPA_COSTS[i] = cum_DAPA_COSTS[i]
            dis_cum_ACE_COSTS[i] = cum_ACE_COSTS[i]
            dis_diagnosis_COSTS_TRACKER[i] = diagnosis_COSTS_TRACKER[i]
            dis_screening_COSTS_TRACKER[i] = screening_COSTS_TRACKER[i]

            dis_LY_detected[i] = (
                sum(trace[detected_or_treated_list].iloc[i]) * CYCLE_LENGTH
            )
            dis_QALY_detected[i] = this_cycle_QALY_detected
            dis_cost_detected[i] = this_cycle_cost_detected

        else:
            discount = 1 / ((1 + discountRates[0]) ** (i - startDiscountCycle + 1))
            dis_LY[i] = LY[i] * discount
            CUM_LY[i] = CUM_LY[i - 1] + LY[i]
            dis_CUM_LY[i] = dis_CUM_LY[i - 1] + dis_LY[i]

            dis_QALY[i] = QALY[i] * discount
            CUM_QALY[i] = CUM_QALY[i - 1] + QALY[i]
            dis_CUM_QALY[i] = dis_CUM_QALY[i - 1] + dis_QALY[i]

            dis_COSTS[i] = COSTS[i] * discount
            CUM_COSTS[i] = CUM_COSTS[i - 1] + COSTS[i]
            dis_CUM_COSTS[i] = dis_CUM_COSTS[i - 1] + dis_COSTS[i]

            dis_cum_DAPA_COSTS[i] = cum_DAPA_COSTS[i] * discount
            dis_cum_ACE_COSTS[i] = cum_ACE_COSTS[i] * discount
            dis_diagnosis_COSTS_TRACKER[i] = diagnosis_COSTS_TRACKER[i] * discount
            dis_screening_COSTS_TRACKER[i] = screening_COSTS_TRACKER[i] * discount

            dis_LY_detected[i] = (
                sum(trace[detected_or_treated_list].iloc[i]) * CYCLE_LENGTH * discount
            )
            dis_QALY_detected[i] = this_cycle_QALY_detected * discount
            dis_cost_detected[i] = this_cycle_cost_detected * discount

    trace["LY"] = pd.Series(LY, index=trace.index)
    trace["cum_LY"] = pd.Series(CUM_LY, index=trace.index)
    trace["dis_LY"] = pd.Series(dis_LY, index=trace.index)
    trace["dis_cum_LY"] = pd.Series(dis_CUM_LY, index=trace.index)

    trace["QALY"] = pd.Series(QALY, index=trace.index)
    trace["cum_QALY"] = pd.Series(CUM_QALY, index=trace.index)
    trace["dis_QALY"] = pd.Series(dis_QALY, index=trace.index)
    trace["dis_cum_QALY"] = pd.Series(dis_CUM_QALY, index=trace.index)

    trace["COSTS"] = pd.Series(COSTS, index=trace.index)
    trace["cum_COSTS"] = pd.Series(CUM_COSTS, index=trace.index)
    trace["dis_COSTS"] = pd.Series(dis_COSTS, index=trace.index)
    trace["dis_cum_COSTS"] = pd.Series(dis_CUM_COSTS, index=trace.index)

    trace["DAPA_COSTS"] = pd.Series(cum_DAPA_COSTS, index=trace.index)
    trace["ACE_COSTS"] = pd.Series(cum_ACE_COSTS, index=trace.index)
    trace["SCREENING_COSTS"] = pd.Series(screening_COSTS_TRACKER, index=trace.index)
    trace["DIAGNOSIS_COSTS"] = pd.Series(diagnosis_COSTS_TRACKER, index=trace.index)

    trace["dis_LY_detected"] = pd.Series(dis_LY_detected, index=trace.index)
    trace["dis_QALY_detected"] = pd.Series(dis_QALY_detected, index=trace.index)
    trace["dis_cost_detected"] = pd.Series(dis_cost_detected, index=trace.index)

    trace["dis_DAPA_COSTS"] = pd.Series(dis_cum_DAPA_COSTS, index=trace.index)
    trace["dis_ACE_COSTS"] = pd.Series(dis_cum_ACE_COSTS, index=trace.index)
    trace["dis_SCREENING_COSTS"] = pd.Series(
        dis_screening_COSTS_TRACKER, index=trace.index
    )
    trace["dis_DIAGNOSIS_COSTS"] = pd.Series(
        dis_diagnosis_COSTS_TRACKER, index=trace.index
    )

    return trace


THESE_STAGES_dir = [
    "No CKD",
    "Stage_1_micro_notdetected",
    "Stage_1_macro_notdetected",
    "Stage_2_no_albumin_notdetected",
    "Stage_2_micro_notdetected",
    "Stage_2_macro_notdetected",
    "Stage_3a_no_albumin_notdetected",
    "Stage_3a_micro_notdetected",
    "Stage_3a_macro_notdetected",
    "Stage_3b_no_albumin_notdetected",
    "Stage_3b_micro_notdetected",
    "Stage_3b_macro_notdetected",
    "Stage_4_no_albumin_notdetected",
    "Stage_4_micro_notdetected",
    "Stage_4_macro_notdetected",
    "Stage_1_micro_detected",
    "Stage_1_macro_detected",
    "Stage_2_micro_detected",
    "Stage_2_macro_detected",
    "Stage_3a_no_albumin_detected",
    "Stage_3a_micro_detected",
    "Stage_3a_macro_detected",
    "Stage_3b_no_albumin_detected",
    "Stage_3b_micro_detected",
    "Stage_3b_macro_detected",
    "Stage_4_no_albumin_detected",
    "Stage_4_micro_detected",
    "Stage_4_macro_detected",
    "Stage_1_micro_detected_treated",
    "Stage_1_macro_detected_treated",
    "Stage_2_micro_detected_treated",
    "Stage_2_macro_detected_treated",
    "Stage_3a_no_albumin_detected_treated",
    "Stage_3a_micro_detected_treated",
    "Stage_3a_macro_detected_treated",
    "Stage_3b_no_albumin_detected_treated",
    "Stage_3b_micro_detected_treated",
    "Stage_3b_macro_detected_treated",
    "Stage_4_no_albumin_detected_treated",
    "Stage_4_micro_detected_treated",
    "Stage_4_macro_detected_treated",
    "KF_PRE_KRT_no_albumin",
    "KF_PRE_KRT_micro",
    "KF_PRE_KRT_macro",
    "KF_no_albumin",
    "KF_micro",
]


THESE_STAGES = [
    "No CKD_notdetected",
    "Stage_1_micro_notdetected",
    "Stage_1_macro_notdetected",
    "Stage_2_no_albumin_notdetected",
    "Stage_2_micro_notdetected",
    "Stage_2_macro_notdetected",
    "Stage_3a_no_albumin_notdetected",
    "Stage_3a_micro_notdetected",
    "Stage_3a_macro_notdetected",
    "Stage_3b_no_albumin_notdetected",
    "Stage_3b_micro_notdetected",
    "Stage_3b_macro_notdetected",
    "Stage_4_no_albumin_notdetected",
    "Stage_4_micro_notdetected",
    "Stage_4_macro_notdetected",
    "No CKD_detected",
    "Stage_1_micro_detected",
    "Stage_1_macro_detected",
    "Stage_2_no_albumin_detected",
    "Stage_2_micro_detected",
    "Stage_2_macro_detected",
    "Stage_3a_no_albumin_detected",
    "Stage_3a_micro_detected",
    "Stage_3a_macro_detected",
    "Stage_3b_no_albumin_detected",
    "Stage_3b_micro_detected",
    "Stage_3b_macro_detected",
    "Stage_4_no_albumin_detected",
    "Stage_4_micro_detected",
    "Stage_4_macro_detected",
    "No CKD_detected_treated",
    "Stage_1_micro_detected_treated",
    "Stage_1_macro_detected_treated",
    "Stage_2_no_albumin_detected_treated",
    "Stage_2_micro_detected_treated",
    "Stage_2_macro_detected_treated",
    "Stage_3a_no_albumin_detected_treated",
    "Stage_3a_micro_detected_treated",
    "Stage_3a_macro_detected_treated",
    "Stage_3b_no_albumin_detected_treated",
    "Stage_3b_micro_detected_treated",
    "Stage_3b_macro_detected_treated",
    "Stage_4_no_albumin_detected_treated",
    "Stage_4_micro_detected_treated",
    "Stage_4_macro_detected_treated",
    "KF_PRE_KRT_no_albumin",
    "KF_PRE_KRT_micro",
    "KF_PRE_KRT_macro",
    "KF_no_albumin",
    "KF_micro",
    "KF_macro",
]

THESE_STAGES2 = [
    "No CKD",
    "Stage_1_micro_notdetected",
    "Stage_1_macro_notdetected",
    "Stage_2_no_albumin_notdetected",
    "Stage_2_micro_notdetected",
    "Stage_2_macro_notdetected",
    "Stage_3a_no_albumin_notdetected",
    "Stage_3a_micro_notdetected",
    "Stage_3a_macro_notdetected",
    "Stage_3b_no_albumin_notdetected",
    "Stage_3b_micro_notdetected",
    "Stage_3b_macro_notdetected",
    "Stage_4_no_albumin_notdetected",
    "Stage_4_micro_notdetected",
    "Stage_4_macro_notdetected",
    "Stage_1_micro_detected",
    "Stage_1_macro_detected",
    "Stage_2_micro_detected",
    "Stage_2_macro_detected",
    "Stage_3a_no_albumin_detected",
    "Stage_3a_micro_detected",
    "Stage_3a_macro_detected",
    "Stage_3b_no_albumin_detected",
    "Stage_3b_micro_detected",
    "Stage_3b_macro_detected",
    "Stage_4_no_albumin_detected",
    "Stage_4_micro_detected",
    "Stage_4_macro_detected",
    "Stage_1_micro_detected_treated",
    "Stage_1_macro_detected_treated",
    "Stage_2_micro_detected_treated",
    "Stage_2_macro_detected_treated",
    "Stage_3a_no_albumin_detected_treated",
    "Stage_3a_micro_detected_treated",
    "Stage_3a_macro_detected_treated",
    "Stage_3b_no_albumin_detected_treated",
    "Stage_3b_micro_detected_treated",
    "Stage_3b_macro_detected_treated",
    "Stage_4_no_albumin_detected_treated",
    "Stage_4_micro_detected_treated",
    "Stage_4_macro_detected_treated",
    "KF_PRE_KRT_no_albumin",
    "KF_PRE_KRT_micro",
    "KF_PRE_KRT_macro",
    "KF_no_albumin",
    "KF_micro",
]
HEALTH_STATES = THESE_STAGES + ["Dead"]

THESE_STAGES2_2021 = []
for j in THESE_STAGES2:
    THESE_STAGES2_2021.append(j + "_2021")

HEALTH_STATES_2021 = []
for j in HEALTH_STATES:
    HEALTH_STATES_2021.append(j + "_2021")

STAGES_WITH_ACE = [
    "No CKD_detected_treated",
    "Stage_1_micro_detected_treated",
    "Stage_1_macro_detected_treated",
    "Stage_2_micro_detected_treated",
    "Stage_2_macro_detected_treated",
    "Stage_2_no_albumin_detected_treated",
    "Stage_3a_no_albumin_detected_treated",
    "Stage_3a_micro_detected_treated",
    "Stage_3a_macro_detected_treated",
    "Stage_3b_no_albumin_detected_treated",
    "Stage_3b_micro_detected_treated",
    "Stage_3b_macro_detected_treated",
    "Stage_4_no_albumin_detected_treated",
    "Stage_4_micro_detected_treated",
    "Stage_4_macro_detected_treated",
    "KF_PRE_KRT_no_albumin",
    "KF_PRE_KRT_micro",
    "KF_PRE_KRT_macro",
]


STAGES_WITH_DAPA = [
    "Stage_2_micro_detected_treated",
    "Stage_2_macro_detected_treated",
    "Stage_3a_micro_detected_treated",
    "Stage_3a_macro_detected_treated",
    "Stage_3b_micro_detected_treated",
    "Stage_3b_macro_detected_treated",
    "Stage_4_micro_detected_treated",
    "Stage_4_macro_detected_treated",
    "KF_PRE_KRT_micro",
    "KF_PRE_KRT_macro",
]


not_detected_cols_list = []
detected_or_treated_cols_list = []
treated_cols_list = []
detected_not_treated_cols_list = []
all_kf_cols_list = []
detected_or_treated_list = []
treated_with_dapa_list = []
for j in range(len(HEALTH_STATES)):
    if "notdetected" in HEALTH_STATES[j]:
        not_detected_cols_list.append(j)
    if (
        "detected" in HEALTH_STATES[j] or "KF" in HEALTH_STATES[j]
    ) and "notdetected" not in HEALTH_STATES[j]:
        detected_or_treated_cols_list.append(j)
        detected_or_treated_list.append(HEALTH_STATES[j])
    if "treated" in HEALTH_STATES[j] or "KF" in HEALTH_STATES[j]:
        treated_cols_list.append(j)
    if HEALTH_STATES[j] in STAGES_WITH_DAPA:
        treated_with_dapa_list.append(j)

CYCLE_LENGTH = 1 / 4.0
parser = ArgumentParser()
parser.add_argument("-d", dest="data_path", required=True, help="data path to read in")
parser.add_argument("-p", dest="PSA_path", required=True, help="data path to PSA")
parser.add_argument("-s", dest="save_path", required=True, help="data path to save to")
parser.add_argument("-r", dest="race_group", required=True, help="sub population")
parser.add_argument("-sa", dest="starting_age", required=True, help="sub population")
parser.add_argument("-si", dest="start_index", required=True, help="sub population")

args = parser.parse_args()
data_p = args.data_path
data_psa = args.PSA_path
save_p = args.save_path
race_g = args.race_group
STARTING_AGE = int(args.starting_age)
START_INDEX = int(args.start_index)
save_p_new = str(START_INDEX) + save_p

if race_g == "NHW":
    male_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicWhiteMale.xlsx", engine="openpyxl"
    )
    female_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicWhiteFemale.xlsx", engine="openpyxl"
    )
elif race_g == "NHB":
    male_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicBlackMale.xlsx", engine="openpyxl"
    )
    female_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicBlackFemale.xlsx", engine="openpyxl"
    )
elif race_g == "H":
    male_life_table = pd.read_excel(
        "2019 Life Tables/HispanicMale.xlsx", engine="openpyxl"
    )
    female_life_table = pd.read_excel(
        "2019 Life Tables/HispanicFemale.xlsx", engine="openpyxl"
    )
elif race_g == "NHA":
    male_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicAsianMale.xlsx", engine="openpyxl"
    )
    female_life_table = pd.read_excel(
        "2019 Life Tables/NonHispanicAsianFemale.xlsx", engine="openpyxl"
    )
else:
    male_life_table = pd.read_excel(
        "2019 Life Tables/OtherMale.xlsx", engine="openpyxl"
    )
    female_life_table = pd.read_excel(
        "2019 Life Tables/OtherFemale.xlsx", engine="openpyxl"
    )

male_life_table = convert_life_cycles(male_life_table)
female_life_table = convert_life_cycles(female_life_table)

DEAD_INDEX = 51

stage_1_age_acceleration_list = [
    0,
    1,
    2,
    15,
    16,
    17,
    30,
    31,
    32,
]  # , 3, 4, 5, 17, 18,30,31]
stage_2_age_acceleration_list = [3, 4, 5, 18, 19, 20, 33, 34, 35]
stage_3_age_acceleration_list = [
    6,
    7,
    8,
    9,
    10,
    11,
    21,
    22,
    23,
    24,
    25,
    26,
    36,
    37,
    38,
    39,
    40,
    41,
]

detected_treated_accelerated = [
    "Stage_1_micro_notdetected",
    "Stage_1_macro_notdetected",
    "Stage_2_micro_notdetected",
    "Stage_2_macro_notdetected",
    "Stage_3a_no_albumin_notdetected",
    "Stage_3a_micro_notdetected",
    "Stage_3a_macro_notdetected",
    "Stage_3b_no_albumin_notdetected",
    "Stage_3b_micro_notdetected",
    "Stage_3b_macro_notdetected",
    "Stage_4_no_albumin_notdetected",
    "Stage_4_micro_notdetected",
    "Stage_4_macro_notdetected",
]

NOT_TRUE_CKD_COLS = ["No CKD_notdetected", "Stage_2_no_albumin_notdetected"]
NOT_TRUE_CKD_COLS_TREATED = [
    "No CKD_detected_treated",
    "Stage_2_no_albumin_detected_treated",
]
alive_states = HEALTH_STATES[:DEAD_INDEX]
NOT_TRUE_CKD_COLS = ["No CKD_notdetected", "Stage_2_no_albumin_notdetected"]
NOT_TRUE_CKD_COLS2 = [
    "No CKD_notdetected",
    "Stage_2_no_albumin_notdetected",
    "Stage_3a_no_albumin_notdetected",
    "Stage_3b_no_albumin_notdetected",
    "Stage_4_no_albumin_notdetected",
]
ABNORMAL_CKD_COLS = [
    "Stage_3a_micro_notdetected",
    "Stage_3a_macro_notdetected",
    "Stage_3b_micro_notdetected",
    "Stage_3b_macro_notdetected",
    "Stage_4_micro_notdetected",
    "Stage_4_macro_notdetected",
]

kf_number_cols_list = [45, 46, 47]

discountRates = [(1 + 0.03) ** (0.25) - 1, 3.0e-4, 3.0e-4]
startDiscountCycle = 1

AGE_LIST = male_life_table["Age"].unique()

male_x = [25, 35, 45, 55, 65, 75, 85]
male_y = [0.934, 0.925, 0.894, 0.870, 0.852, 0.816, 0.807]
male_QALY = interp1d(male_x, male_y, fill_value="extrapolate", kind="linear")
female_x = [25, 35, 45, 55, 65, 75, 85]
female_y = [0.920, 0.900, 0.871, 0.846, 0.822, 0.784, 0.747]
female_QALY = interp1d(female_x, female_y, fill_value="extrapolate", kind="linear")
age_qaly_male = [0 for i in range(len(male_life_table))]
age_qaly_female = [0 for i in range(len(female_life_table))]
for i in range(len(male_life_table)):
    age_qaly_male[i] = round(min(1.0, male_QALY(i).item()), 3)
    age_qaly_female[i] = round(min(1.0, female_QALY(i).item()), 3)
male_life_table["age_qaly"] = pd.Series(age_qaly_male, index=male_life_table.index)
female_life_table["age_qaly"] = pd.Series(
    age_qaly_female, index=female_life_table.index
)


MEPS_data = pd.read_excel("MEPS_data.xlsx", engine="openpyxl")
male_baseline = interp1d(
    MEPS_data["Age"], MEPS_data["Male-2021"], fill_value="extrapolate", kind="linear"
)
# kind='linear')
female_baseline = interp1d(
    MEPS_data["Age"], MEPS_data["Female-2021"], fill_value="extrapolate", kind="linear"
)
male_cost = []
female_cost = []
male_cost2 = []
female_cost2 = []
for j in range(86, 101):
    male_cost.append(male_baseline(j).item())
    female_cost.append(female_baseline(j).item())
    male_cost2.append(0)
    female_cost2.append(0)

add_to_MEPS = pd.DataFrame(range(86, 101), columns=["Age"])
add_to_MEPS["Female"] = female_cost2
add_to_MEPS["Male"] = male_cost2
add_to_MEPS["Female-2021"] = female_cost
add_to_MEPS["Male-2021"] = male_cost
MEPS_data = pd.concat([MEPS_data, add_to_MEPS])
MEPS_data = MEPS_data.reset_index()


start = time.time()

SIR_parameters_2021 = pd.read_excel(f"SIR_PSA_inputs_race_ethnic_groups/{data_p}", engine="openpyxl")
PSA_df = pd.read_excel(f"SIR_PSA_inputs_race_ethnic_groups/{data_psa}", engine="openpyxl")
results_df = pd.concat([SIR_parameters_2021, PSA_df], axis=1)

if STARTING_AGE == 35:
    SCREENING_AFTER_OPTIONS = [
        [0],
        [0, 10],
        [0, 10, 20],
        [0, 10, 20, 30],
        [0, 10, 20, 30, 40],
        [0, 5, 10],
        [0, 5, 10, 15, 20],
        [0, 5, 10, 15, 20, 25, 30],
        [0, 5, 10, 15, 20, 25, 30, 35, 40],
        [10],
        [10, 20],
        [10, 20, 30],
        [10, 20, 30, 40],
        [10, 15, 20],
        [10, 15, 20, 25, 30],
        [10, 15, 20, 25, 30, 35, 40],
        [20],
        [
            20,
            30,
        ],
        [20, 30, 40],
        [20, 25, 30],
        [20, 25, 30, 35, 40],
        [30],
        [30, 40],
        [30, 35, 40],
        [40],
    ]
elif STARTING_AGE == 45:
    SCREENING_AFTER_OPTIONS = [
        [0],
        [0, 10],
        [0, 10, 20],
        [0, 10, 20, 30],
        [0, 5, 10],
        [0, 5, 10, 15, 20],
        [0, 5, 10, 15, 20, 25, 30],
        [10],
        [10, 20],
        [10, 20, 30],
        [10, 15, 20],
        [10, 15, 20, 25, 30],
        [20],
        [20, 30],
        [20, 25, 30],
        [30],
    ]
elif STARTING_AGE == 55:
    SCREENING_AFTER_OPTIONS = [
        [0],
        [0, 10],
        [0, 10, 20],
        [0, 5, 10],
        [0, 5, 10, 15, 20],
        [10],
        [10, 20],
        [10, 15, 20],
        [20],
    ]
elif STARTING_AGE == 65:
    SCREENING_AFTER_OPTIONS = [[0], [0, 10], [0, 5, 10], [10]]
else:
    SCREENING_AFTER_OPTIONS = [[0]]


count = 0
for j in range(START_INDEX, START_INDEX + 1000):
    print(j)

    this_index = j

    eGFR_PROBS = dict()
    for s in HEALTH_STATES:
        if "No CKD" in s or "Stage_1" in s:
            eGFR_PROBS[s] = results_df["eGFR_stage_1"].iloc[this_index]
        if "Stage_2" in s:
            eGFR_PROBS[s] = results_df["eGFR_stage_2"].iloc[this_index]
        if "Stage_3a" in s or "Stage_3b" in s or "Stage_4" in s:
            eGFR_PROBS[s] = results_df["eGFR_stage_3a"].iloc[this_index]
        if "KF_PRE_KRT" in s:
            eGFR_PROBS[s] = results_df["eGFR_stage_5"].iloc[this_index]

    eGFR_PROBS["STAGE_1_OLDER"] = results_df["eGFR_stage_1_older"].iloc[this_index]
    eGFR_PROBS["STAGE_2_OLDER"] = results_df["eGFR_stage_2_older"].iloc[this_index]
    eGFR_PROBS["STAGE_3_OLDER"] = results_df["eGFR_stage_3_older"].iloc[this_index]

    YOUNG_AGE = results_df["young_age"].iloc[this_index]
    OLD_AGE = results_df["old_age"].iloc[this_index]
    AGE_ACCELERATION_LIST = [YOUNG_AGE, OLD_AGE]

    STAGE_1_ACCELERATION_LINEAR = interp1d(
        AGE_ACCELERATION_LIST,
        [
            results_df["eGFR_stage_1"].iloc[this_index],
            results_df["eGFR_stage_1_older"].iloc[this_index],
        ],
        fill_value="extrapolate",
        kind="linear",
    )

    STAGE_2_ACCELERATION_LINEAR = interp1d(
        AGE_ACCELERATION_LIST,
        [
            results_df["eGFR_stage_2"].iloc[this_index],
            results_df["eGFR_stage_2_older"].iloc[this_index],
        ],
        fill_value="extrapolate",
        kind="linear",
    )

    STAGE_3_ACCELERATION_LINEAR = interp1d(
        AGE_ACCELERATION_LIST,
        [
            results_df["eGFR_stage_3a"].iloc[this_index],
            results_df["eGFR_stage_3_older"].iloc[this_index],
        ],
        fill_value="extrapolate",
        kind="linear",
    )

    ALBUMIN_ACCELERATION = dict()
    for s in HEALTH_STATES:
        if "No CKD" in s or "no_albumin" in s:
            ALBUMIN_ACCELERATION[s] = 1
        if "micro" in s:
            ALBUMIN_ACCELERATION[s] = results_df["micro_acceleration"].iloc[this_index]
        if "macro" in s:
            ALBUMIN_ACCELERATION[s] = results_df["macro_acceleration"].iloc[this_index]

    DETECTION_ACCELERATION = dict()
    for s in HEALTH_STATES:
        if "No CKD" in s:
            DETECTION_ACCELERATION[s] = 0
        if "Stage_1" in s:
            DETECTION_ACCELERATION[s] = results_df["detection_stage_1"].iloc[this_index]
        if "Stage_2_no_albumin" in s:
            DETECTION_ACCELERATION[s] = 0
        if "Stage_2" in s and "no_albumin" not in s:
            DETECTION_ACCELERATION[s] = results_df["detection_stage_2"].iloc[this_index]
        if "Stage_3a" in s or "Stage_3b" in s:
            DETECTION_ACCELERATION[s] = results_df["detection_stage_3"].iloc[this_index]
        if "Stage_4" in s:
            DETECTION_ACCELERATION[s] = results_df["detection_stage_4"].iloc[this_index]

    DETECTION_ACCELERATION_AGE = int(
        results_df["detection_acceleration_age"].iloc[this_index]
    )
    DETECTION_ACCELERATION_HR = results_df["detection_acceleration_HR"].iloc[this_index]

    TREATMENT_ACCELERATION_AGE = int(
        results_df["treatment_acceleration_age"].iloc[this_index]
    )
    TREATMENT_ACCELERATION_HR = results_df["treatment_acceleration_HR"].iloc[this_index]

    ALBUMIN_PROBS = dict()
    for s in HEALTH_STATES:
        if "No CKD" in s or "Stage_1" in s or "Stage_2" in s:
            ALBUMIN_PROBS[s] = results_df["albumin_transition_1"].iloc[this_index]
        if "Stage_3a" in s or "Stage_3b" in s:
            ALBUMIN_PROBS[s] = results_df["albumin_transition_3"].iloc[this_index]
        if "Stage_4" in s:
            ALBUMIN_PROBS[s] = results_df["albumin_transition_4"].iloc[this_index]
        if "KF_PRE_KRT" in s or "KF" in s:
            ALBUMIN_PROBS[s] = results_df["albumin_transition_kf"].iloc[this_index]

    IMMEDIATE_TREATMENT = dict()
    for s in HEALTH_STATES:
        if "No CKD" in s:
            IMMEDIATE_TREATMENT[s] = 0
        if "Stage_1" in s:
            IMMEDIATE_TREATMENT[s] = results_df["treatment_stage_1"].iloc[this_index]
        if "Stage_2" in s:
            IMMEDIATE_TREATMENT[s] = results_df["treatment_stage_2"].iloc[this_index]
        if "Stage_3a" in s or "Stage_3b" in s:
            IMMEDIATE_TREATMENT[s] = results_df["treatment_stage_3"].iloc[this_index]
        if "Stage_4" in s:
            IMMEDIATE_TREATMENT[s] = results_df["treatment_stage_4"].iloc[this_index]

    PROB_DISCONTINUE = results_df["discontinue"].iloc[this_index]

    PROB_DISCONTINUE2 = results_df["SGLT2_discontinue"].iloc[this_index]

    TREATED_PROB = results_df["treatment"].iloc[this_index]

    CYCLE_LENGTH = 1 / 4

    STARTING_AGE = 35
    this_dir_2021 = np.array(results_df[THESE_STAGES2_2021].iloc[this_index])
    this_starting_distribution = [0 for l in HEALTH_STATES]
    for l in range(len(THESE_STAGES2_2021)):
        if THESE_STAGES2_2021[l] == "No CKD_2021":
            this_starting_distribution[0] = this_dir_2021[l]
        elif THESE_STAGES2_2021[l] == "Stage_2_no_albumin_notdetected_2021":
            this_starting_distribution[3] = this_dir_2021[l]
        else:
            state_index = HEALTH_STATES_2021.index(THESE_STAGES2_2021[l])
            this_starting_distribution[state_index] = this_dir_2021[l]

    if sum(this_starting_distribution) < 1:
        this_starting_distribution[50] = 1 - sum(this_starting_distribution)

    SCREENING_SPECIFICITY = results_df["screening_sensitivity"].iloc[this_index]
    SCREENING_SENSITIVITY = results_df["screening_specificity"].iloc[this_index]

    ADVERSE_PERCENTAGE = results_df["adverse_event_prop"].iloc[this_index]
    ADVERSE_COST = results_df["disutility_medication_cost"].iloc[this_index]
    ADVERSE_QALY = -results_df["disutility_medication_decrement"].iloc[this_index]
    TREATMENT_INITATION = (
        results_df["TREATMENT ADHERENCE"].iloc[this_index] - ADVERSE_PERCENTAGE
    )

    UTI_EVENT_COST = results_df["UTI_event_cost"].iloc[this_index]
    UTI_EVENT_PROP = results_df["UTI_event_prop"].iloc[this_index]
    UTI_EVENT_QALY = -results_df["disutility_UTI_decrement"].iloc[this_index]

    DKA_EVENT_COST = results_df["DKA_event_cost"].iloc[this_index]
    DKA_EVENT_PROP = results_df["DKA_event_prop"].iloc[this_index]
    DKA_EVENT_QALY = -results_df["disutility_DKA_decrement"].iloc[this_index]

    SCREENING_COST = results_df["screening_cost"].iloc[this_index]
    SERUM_CR = results_df["eGFR_cost"].iloc[this_index]
    ULTRASOUND = results_df["ultrasound_cost"].iloc[this_index]
    OFFICE_VISIT = 0  

    HAZARDS_RATIOS = dict()

    HAZARDS_RATIOS["No CKD_notdetected"] = 1.00
    HAZARDS_RATIOS["No CKD_detected"] = 1.00
    HAZARDS_RATIOS["No CKD_detected_treated"] = 1.00

    HAZARDS_RATIOS["Stage_1_micro_notdetected"] = 1.0
    HAZARDS_RATIOS["Stage_1_macro_notdetected"] = 1.0
    HAZARDS_RATIOS["Stage_1_micro_detected"] = HAZARDS_RATIOS[
        "Stage_1_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_1_macro_detected"] = HAZARDS_RATIOS[
        "Stage_1_macro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_1_micro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_1_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_1_macro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_1_macro_notdetected"
    ]

    HAZARDS_RATIOS["Stage_2_no_albumin_notdetected"] = 1.0
    HAZARDS_RATIOS["Stage_2_micro_notdetected"] = 1.0
    HAZARDS_RATIOS["Stage_2_macro_notdetected"] = 1.0
    HAZARDS_RATIOS["Stage_2_no_albumin_detected"] = HAZARDS_RATIOS[
        "Stage_2_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_2_micro_detected"] = HAZARDS_RATIOS[
        "Stage_2_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_2_macro_detected"] = HAZARDS_RATIOS[
        "Stage_2_macro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_2_no_albumin_detected_treated"] = HAZARDS_RATIOS[
        "Stage_2_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_2_micro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_2_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_2_macro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_2_macro_notdetected"
    ]

    HAZARDS_RATIOS["Stage_3a_no_albumin_notdetected"] = results_df[
        "Stage3a mortality"
    ].iloc[this_index]
    HAZARDS_RATIOS["Stage_3a_micro_notdetected"] = results_df["Stage3a mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_3a_macro_notdetected"] = results_df["Stage3a mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_3a_no_albumin_detected"] = HAZARDS_RATIOS[
        "Stage_3a_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3a_micro_detected"] = HAZARDS_RATIOS[
        "Stage_3a_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3a_macro_detected"] = HAZARDS_RATIOS[
        "Stage_3a_macro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3a_no_albumin_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3a_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3a_micro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3a_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3a_macro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3a_macro_notdetected"
    ]

    HAZARDS_RATIOS["Stage_3b_no_albumin_notdetected"] = results_df[
        "Stage3b mortality"
    ].iloc[this_index]
    HAZARDS_RATIOS["Stage_3b_micro_notdetected"] = results_df["Stage3b mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_3b_macro_notdetected"] = results_df["Stage3b mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_3b_no_albumin_detected"] = HAZARDS_RATIOS[
        "Stage_3b_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3b_micro_detected"] = HAZARDS_RATIOS[
        "Stage_3b_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3b_macro_detected"] = HAZARDS_RATIOS[
        "Stage_3b_macro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3b_no_albumin_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3b_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3b_micro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3b_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_3b_macro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_3b_macro_notdetected"
    ]

    HAZARDS_RATIOS["Stage_4_no_albumin_notdetected"] = results_df[
        "Stage4 mortality"
    ].iloc[this_index]
    HAZARDS_RATIOS["Stage_4_micro_notdetected"] = results_df["Stage4 mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_4_macro_notdetected"] = results_df["Stage4 mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["Stage_4_no_albumin_detected"] = HAZARDS_RATIOS[
        "Stage_4_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_4_micro_detected"] = HAZARDS_RATIOS[
        "Stage_4_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_4_macro_detected"] = HAZARDS_RATIOS[
        "Stage_4_macro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_4_no_albumin_detected_treated"] = HAZARDS_RATIOS[
        "Stage_4_no_albumin_notdetected"
    ]
    HAZARDS_RATIOS["Stage_4_micro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_4_micro_notdetected"
    ]
    HAZARDS_RATIOS["Stage_4_macro_detected_treated"] = HAZARDS_RATIOS[
        "Stage_4_macro_notdetected"
    ]

    HAZARDS_RATIOS["KF_PRE_KRT_no_albumin"] = results_df["Stage4 mortality"].iloc[
        this_index
    ]
    HAZARDS_RATIOS["KF_PRE_KRT_micro"] = results_df["Stage4 mortality"].iloc[this_index]
    HAZARDS_RATIOS["KF_PRE_KRT_macro"] = results_df["Stage4 mortality"].iloc[this_index]
    HAZARDS_RATIOS["KF_no_albumin"] = results_df["Stage5 mortality"].iloc[this_index]
    HAZARDS_RATIOS["KF_micro"] = results_df["Stage5 mortality"].iloc[this_index]
    HAZARDS_RATIOS["KF_macro"] = results_df["Stage5 mortality"].iloc[this_index]

    STAGE_QALYS = dict()
    STAGE_QALYS["No CKD_notdetected"] = 1
    STAGE_QALYS["Stage_1_micro_detected"] = 1
    STAGE_QALYS["Stage_1_macro_detected"] = 1
    STAGE_QALYS["No CKD_detected"] = 1
    STAGE_QALYS["Stage_1_micro_detected_treated"] = 1
    STAGE_QALYS["Stage_1_macro_detected_treated"] = 1
    STAGE_QALYS["No CKD_detected_treated"] = 1
    STAGE_QALYS["Stage_1_micro_notdetected"] = 1
    STAGE_QALYS["Stage_1_macro_notdetected"] = 1

    STAGE_QALYS["Stage_2_micro_detected"] = results_df["Stage2_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_2_macro_detected"] = results_df["Stage2_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_2_micro_detected_treated"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_macro_detected_treated"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_no_albumin_notdetected"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_no_albumin_detected"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_no_albumin_detected_treated"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_micro_notdetected"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_2_macro_notdetected"] = results_df["Stage2_QALY"].iloc[
        this_index
    ]

    STAGE_QALYS["Stage_3a_no_albumin_detected"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3a_micro_detected"] = results_df["Stage3a_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_3a_macro_detected"] = results_df["Stage3a_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_3a_no_albumin_detected_treated"] = results_df[
        "Stage3a_QALY"
    ].iloc[this_index]
    STAGE_QALYS["Stage_3a_micro_detected_treated"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3a_macro_detected_treated"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3a_no_albumin_notdetected"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3a_micro_notdetected"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3a_macro_notdetected"] = results_df["Stage3a_QALY"].iloc[
        this_index
    ]

    STAGE_QALYS["Stage_3b_no_albumin_detected"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3b_micro_detected"] = results_df["Stage3b_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_3b_macro_detected"] = results_df["Stage3b_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_3b_no_albumin_detected_treated"] = results_df[
        "Stage3b_QALY"
    ].iloc[this_index]
    STAGE_QALYS["Stage_3b_micro_detected_treated"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3b_macro_detected_treated"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]

    STAGE_QALYS["Stage_3b_no_albumin_notdetected"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3b_micro_notdetected"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_3b_macro_notdetected"] = results_df["Stage3b_QALY"].iloc[
        this_index
    ]

    STAGE_QALYS["Stage_4_no_albumin_detected"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_micro_detected"] = results_df["Stage4_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_4_macro_detected"] = results_df["Stage4_QALY"].iloc[this_index]
    STAGE_QALYS["Stage_4_no_albumin_detected_treated"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_micro_detected_treated"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_macro_detected_treated"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_no_albumin_notdetected"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_micro_notdetected"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["Stage_4_macro_notdetected"] = results_df["Stage4_QALY"].iloc[
        this_index
    ]

    STAGE_QALYS["KF_PRE_KRT_no_albumin"] = results_df["KF_PRE_KRT_QALY"].iloc[
        this_index
    ]
    STAGE_QALYS["KF_PRE_KRT_micro"] = results_df["KF_PRE_KRT_QALY"].iloc[this_index]
    STAGE_QALYS["KF_PRE_KRT_macro"] = results_df["KF_PRE_KRT_QALY"].iloc[this_index]
    STAGE_QALYS["KF_no_albumin"] = results_df["KF_ON_KRT_QALY"].iloc[this_index]
    STAGE_QALYS["KF_micro"] = results_df["KF_ON_KRT_QALY"].iloc[this_index]
    STAGE_QALYS["KF_macro"] = results_df["KF_ON_KRT_QALY"].iloc[this_index]
    STAGE_QALYS["Dead"] = 0

    ACE_COSTS = results_df["ACE cost"].iloc[this_index] * 3
    DAPA_COST = results_df["dapa cost"].iloc[this_index] * 3
    STAGE_COSTS = dict()
    STAGE_COSTS["No CKD_notdetected"] = 0
    STAGE_COSTS["No CKD_detected"] = 0
    STAGE_COSTS["No CKD_detected_treated"] = 0
    STAGE_COSTS["Stage_1_micro_detected"] = 0
    STAGE_COSTS["Stage_1_macro_detected"] = 0
    STAGE_COSTS["Stage_1_micro_detected_treated"] = 0
    STAGE_COSTS["Stage_1_macro_detected_treated"] = 0
    STAGE_COSTS["Stage_1_micro_notdetected"] = 0
    STAGE_COSTS["Stage_1_macro_notdetected"] = 0

    STAGE_COSTS["Stage_2_micro_detected"] = 0
    STAGE_COSTS["Stage_2_macro_detected"] = 0
    STAGE_COSTS["Stage_2_micro_detected_treated"] = 0
    STAGE_COSTS["Stage_2_macro_detected_treated"] = 0

    STAGE_COSTS["Stage_2_no_albumin_notdetected"] = 0
    STAGE_COSTS["Stage_2_no_albumin_detected"] = 0
    STAGE_COSTS["Stage_2_no_albumin_detected_treated"] = 0
    STAGE_COSTS["Stage_2_micro_notdetected"] = 0
    STAGE_COSTS["Stage_2_macro_notdetected"] = 0

    STAGE_COSTS["Stage_3a_no_albumin_detected"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3a_micro_detected"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3a_macro_detected"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3a_no_albumin_detected_treated"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3a_micro_detected_treated"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3a_macro_detected_treated"] = (
        results_df["Stage3a_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3a_ckd_prev"].iloc[this_index])
        + results_df["Stage3a_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3a_no_albumin_notdetected"] = (
        results_df["Stage3a_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3a_micro_notdetected"] = (
        results_df["Stage3a_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3a_macro_notdetected"] = (
        results_df["Stage3a_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3a_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_no_albumin_detected"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_micro_detected"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_macro_detected"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_no_albumin_detected_treated"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_micro_detected_treated"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_macro_detected_treated"] = (
        results_df["Stage3b_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage3b_ckd_prev"].iloc[this_index])
        + results_df["Stage3b_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_3b_no_albumin_notdetected"] = (
        results_df["Stage3b_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3b_micro_notdetected"] = (
        results_df["Stage3b_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_3b_macro_notdetected"] = (
        results_df["Stage3b_dm_only_cost"].iloc[this_index]
        * (results_df["Stage3b_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_4_no_albumin_detected_treated"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_4_micro_detected_treated"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_4_macro_detected_treated"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_4_no_albumin_detected"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_4_micro_detected"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_4_macro_detected"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["Stage_4_no_albumin_notdetected"] = (
        results_df["Stage4_dm_only_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_4_micro_notdetected"] = (
        results_df["Stage4_dm_only_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["Stage_4_macro_notdetected"] = (
        results_df["Stage4_dm_only_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3

    STAGE_COSTS["KF_PRE_KRT_no_albumin"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["KF_PRE_KRT_micro"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["KF_PRE_KRT_macro"] = (
        results_df["Stage4_ckd_only_cost"].iloc[this_index]
        * (1 - results_df["Stage4_ckd_prev"].iloc[this_index])
        + results_df["Stage4_ckd_dm_cost"].iloc[this_index]
        * (results_df["Stage4_ckd_prev"].iloc[this_index])
    ) * 3
    STAGE_COSTS["KF_no_albumin"] = (
        results_df["Stage5_ckd_only_cost"].iloc[this_index] * 3
    )
    STAGE_COSTS["KF_micro"] = results_df["Stage5_ckd_only_cost"].iloc[this_index] * 3
    STAGE_COSTS["KF_macro"] = results_df["Stage5_ckd_only_cost"].iloc[this_index] * 3
    STAGE_COSTS["Dead"] = 0

    BASELINE_MULTIPLIER = results_df["baseline_costs"].iloc[this_index]
    FEMALE_PREV = 1 - results_df["male_prop"].iloc[this_index]

    eGFR_REDUCTION_DAPA_dict = dict()
    eGFR_REDUCTION_DAPA_dict[39] = min(
        1,
        results_df["diabetic_prevalence_30_39"].iloc[this_index]
        * (results_df["diabetic kidney HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_30_39"].iloc[this_index])
        * (results_df["non diabetic kidney HR"].iloc[this_index]),
    )

    eGFR_REDUCTION_DAPA_dict[49] = min(
        1,
        results_df["diabetic_prevalence_40_49"].iloc[this_index]
        * (results_df["diabetic kidney HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_40_49"].iloc[this_index])
        * (results_df["non diabetic kidney HR"].iloc[this_index]),
    )

    eGFR_REDUCTION_DAPA_dict[59] = min(
        1,
        results_df["diabetic_prevalence_50_59"].iloc[this_index]
        * (results_df["diabetic kidney HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_50_59"].iloc[this_index])
        * (results_df["non diabetic kidney HR"].iloc[this_index]),
    )

    eGFR_REDUCTION_DAPA_dict[69] = min(
        1,
        results_df["diabetic_prevalence_60_69"].iloc[this_index]
        * (results_df["diabetic kidney HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_60_69"].iloc[this_index])
        * (results_df["non diabetic kidney HR"].iloc[this_index]),
    )

    eGFR_REDUCTION_DAPA_dict[70] = min(
        1,
        results_df["diabetic_prevalence_70_79"].iloc[this_index]
        * (results_df["diabetic kidney HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_70_79"].iloc[this_index])
        * (results_df["non diabetic kidney HR"].iloc[this_index]),
    )

    eGFR_REDUCTION_ACE = min(1, results_df["eGFR_reduction"].iloc[this_index])

    ACM_REDUCTION_dict = dict()
    ACM_REDUCTION_dict[39] = min(
        1,
        results_df["diabetic_prevalence_30_39"].iloc[this_index]
        * (results_df["diabetic ACM HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_30_39"].iloc[this_index])
        * (results_df["non diabetic ACM HR"].iloc[this_index]),
    )

    ACM_REDUCTION_dict[49] = min(
        1,
        results_df["diabetic_prevalence_40_49"].iloc[this_index]
        * (results_df["diabetic ACM HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_40_49"].iloc[this_index])
        * (results_df["non diabetic ACM HR"].iloc[this_index]),
    )

    ACM_REDUCTION_dict[59] = min(
        1,
        results_df["diabetic_prevalence_50_59"].iloc[this_index]
        * (results_df["diabetic ACM HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_50_59"].iloc[this_index])
        * (results_df["non diabetic ACM HR"].iloc[this_index]),
    )

    ACM_REDUCTION_dict[69] = min(
        1,
        results_df["diabetic_prevalence_60_69"].iloc[this_index]
        * (results_df["diabetic ACM HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_60_69"].iloc[this_index])
        * (results_df["non diabetic ACM HR"].iloc[this_index]),
    )

    ACM_REDUCTION_dict[70] = min(
        1,
        results_df["diabetic_prevalence_70_79"].iloc[this_index]
        * (results_df["diabetic ACM HR"].iloc[this_index])
        + (1 - results_df["diabetic_prevalence_70_79"].iloc[this_index])
        * (results_df["non diabetic ACM HR"].iloc[this_index]),
    )

    STARTING_AGE_BASELINE = 35
    NUM_ITERATIONS = int((100 - STARTING_AGE_BASELINE) / CYCLE_LENGTH)
    placebo_male_trace_baseline = run_model(True, False, this_starting_distribution)
    placebo_female_trace_baseline = run_model(False, False, this_starting_distribution)

    STARTING_AGE = int(args.starting_age)

    NUM_ITERATIONS = int((100 - STARTING_AGE) / CYCLE_LENGTH)

    if STARTING_AGE == 35:
        placebo_male_trace = placebo_male_trace_baseline
        placebo_female_trace = placebo_female_trace_baseline

    if STARTING_AGE == 45:
        yr_45_yo_m = list(placebo_male_trace_baseline[HEALTH_STATES].iloc[40])
        yr_45_yo_f = list(placebo_female_trace_baseline[HEALTH_STATES].iloc[40])
        yr_45_yo_m = [x / (1 - yr_45_yo_m[-1]) for x in yr_45_yo_m]
        yr_45_yo_f = [x / (1 - yr_45_yo_f[-1]) for x in yr_45_yo_f]
        yr_45_yo_m[DEAD_INDEX] = 0
        yr_45_yo_f[DEAD_INDEX] = 0

        placebo_male_trace = run_model(True, False, yr_45_yo_m)
        placebo_female_trace = run_model(False, False, yr_45_yo_f)

    if STARTING_AGE == 55:
        yr_55_yo_m = list(placebo_male_trace_baseline[HEALTH_STATES].iloc[80])
        yr_55_yo_f = list(placebo_female_trace_baseline[HEALTH_STATES].iloc[80])
        yr_55_yo_m = [x / (1 - yr_55_yo_m[-1]) for x in yr_55_yo_m]
        yr_55_yo_f = [x / (1 - yr_55_yo_f[-1]) for x in yr_55_yo_f]
        yr_55_yo_m[DEAD_INDEX] = 0
        yr_55_yo_f[DEAD_INDEX] = 0

        placebo_male_trace = run_model(True, False, yr_55_yo_m)
        placebo_female_trace = run_model(False, False, yr_55_yo_f)

    if STARTING_AGE == 65:
        yr_65_yo_m = list(placebo_male_trace_baseline[HEALTH_STATES].iloc[120])
        yr_65_yo_f = list(placebo_female_trace_baseline[HEALTH_STATES].iloc[120])
        yr_65_yo_m = [x / (1 - yr_65_yo_m[-1]) for x in yr_65_yo_m]
        yr_65_yo_f = [x / (1 - yr_65_yo_f[-1]) for x in yr_65_yo_f]
        yr_65_yo_m[DEAD_INDEX] = 0
        yr_65_yo_f[DEAD_INDEX] = 0

        placebo_male_trace = run_model(True, False, yr_65_yo_m)
        placebo_female_trace = run_model(False, False, yr_65_yo_f)

    if STARTING_AGE == 75:
        yr_75_yo_m = list(placebo_male_trace_baseline[HEALTH_STATES].iloc[160])
        yr_75_yo_f = list(placebo_female_trace_baseline[HEALTH_STATES].iloc[160])
        yr_75_yo_m = [x / (1 - yr_75_yo_m[-1]) for x in yr_75_yo_m]
        yr_75_yo_f = [x / (1 - yr_75_yo_f[-1]) for x in yr_75_yo_f]
        yr_75_yo_m[DEAD_INDEX] = 0
        yr_75_yo_f[DEAD_INDEX] = 0

        placebo_male_trace = run_model(True, False, yr_75_yo_m)
        placebo_female_trace = run_model(False, False, yr_75_yo_f)

    trace_results_tracker = []

    placebo_male_trace = transform_distributions(placebo_male_trace, True, False)
    placebo_female_trace = transform_distributions(placebo_female_trace, False, False)
    placebo_female_trace["treated with dapa"] = placebo_female_trace[
        STAGES_WITH_DAPA
    ].sum(axis=1) / (1 - placebo_female_trace["Dead"])
    placebo_male_trace["treated with dapa"] = placebo_male_trace[STAGES_WITH_DAPA].sum(
        axis=1
    ) / (1 - placebo_male_trace["Dead"])

    this_tr_arr = []
    this_tr_arr.append(results_df["index"].iloc[j])
    this_tr_arr.append(STARTING_AGE)
    this_tr_arr.append("Placebo")
    this_tr_arr.append(
        placebo_male_trace["dis_cum_LY"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_cum_LY"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["cum_LY"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["cum_LY"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_cum_QALY"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_cum_QALY"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["cum_QALY"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["cum_QALY"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_cum_COSTS"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_cum_COSTS"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["cum_COSTS"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
        + placebo_female_trace["cum_COSTS"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["kf_incidence"].sum() * 100 * (1 - FEMALE_PREV)
        + placebo_female_trace["kf_incidence"].sum() * 100 * (FEMALE_PREV)
    )

    this_tr_arr.append(
        placebo_male_trace["dis_ACE_COSTS"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_ACE_COSTS"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_DAPA_COSTS"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_DAPA_COSTS"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["SCREENING_COSTS"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["SCREENING_COSTS"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_SCREENING_COSTS"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_SCREENING_COSTS"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_DIAGNOSIS_COSTS"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_DIAGNOSIS_COSTS"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(SCREENING_COST + OFFICE_VISIT)
    this_tr_arr.append(
        np.average(
            placebo_male_trace["treated with dapa"],
            weights=(1 - placebo_male_trace["Dead"]),
        )
        * (1 - FEMALE_PREV)
        + np.average(
            placebo_female_trace["treated with dapa"],
            weights=(1 - placebo_female_trace["Dead"]),
        )
        * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_LY_detected"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_LY_detected"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_QALY_detected"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_QALY_detected"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        placebo_male_trace["dis_cost_detected"].sum() * (1 - FEMALE_PREV)
        + placebo_female_trace["dis_cost_detected"].sum() * (FEMALE_PREV)
    )
    this_tr_arr.append(
        (
            placebo_male_trace[detected_or_treated_list].iloc[0].sum()
            + placebo_male_trace["detected_percent"].sum()
        )
        * (1 - FEMALE_PREV)
        + (
            placebo_female_trace[detected_or_treated_list].iloc[0].sum()
            + placebo_female_trace["detected_percent"].sum()
        )
        * (FEMALE_PREV)
    )
    this_tr_arr.append(placebo_male_trace[STAGES_WITH_ACE].iloc[0].sum())

    trace_results_tracker.append(this_tr_arr)

    for scr_option in SCREENING_AFTER_OPTIONS:
        SCREENING_AFTER = scr_option

        CAN_TEST = True
        for s in SCREENING_AFTER:
            if STARTING_AGE + s >= 76:
                CAN_TEST = False

        if CAN_TEST == True:
            SCREENING_AFTER_ITERATIONS = [max(x * 4 - 1, 0) for x in SCREENING_AFTER]
            SCREENING_AFTER_ITERATIONS_C = [max(x * 4, 0) for x in SCREENING_AFTER]

            if STARTING_AGE == 35:
                this_start_m = list(placebo_male_trace_baseline[HEALTH_STATES].iloc[0])
                this_start_f = list(
                    placebo_female_trace_baseline[HEALTH_STATES].iloc[0]
                )
            if STARTING_AGE == 45:
                this_start_m = yr_45_yo_m
                this_start_f = yr_45_yo_f
            if STARTING_AGE == 55:
                this_start_m = yr_55_yo_m
                this_start_f = yr_55_yo_f
            if STARTING_AGE == 65:
                this_start_m = yr_65_yo_m
                this_start_f = yr_65_yo_f
            if STARTING_AGE == 75:
                this_start_m = yr_75_yo_m
                this_start_f = yr_75_yo_f

            screening_male_trace = run_model(True, True, this_start_m)
            screening_female_trace = run_model(False, True, this_start_f)

            screening_male_trace = transform_distributions(
                screening_male_trace, True, True
            )
            screening_female_trace = transform_distributions(
                screening_female_trace, False, True
            )

            screening_female_trace["treated with dapa"] = screening_female_trace[
                STAGES_WITH_DAPA
            ].sum(axis=1) / (1 - screening_female_trace["Dead"])
            screening_male_trace["treated with dapa"] = screening_male_trace[
                STAGES_WITH_DAPA
            ].sum(axis=1) / (1 - screening_male_trace["Dead"])

            this_tr_arr = []
            this_tr_arr.append(results_df["index"].iloc[j])
            this_tr_arr.append(STARTING_AGE)
            this_tr_arr.append(str(scr_option))
            this_tr_arr.append(
                screening_male_trace["dis_cum_LY"].iloc[NUM_ITERATIONS]
                * (1 - FEMALE_PREV)
                + screening_female_trace["dis_cum_LY"].iloc[NUM_ITERATIONS]
                * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["cum_LY"].iloc[NUM_ITERATIONS] * (1 - FEMALE_PREV)
                + screening_female_trace["cum_LY"].iloc[NUM_ITERATIONS] * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_cum_QALY"].iloc[NUM_ITERATIONS]
                * (1 - FEMALE_PREV)
                + screening_female_trace["dis_cum_QALY"].iloc[NUM_ITERATIONS]
                * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["cum_QALY"].iloc[NUM_ITERATIONS]
                * (1 - FEMALE_PREV)
                + screening_female_trace["cum_QALY"].iloc[NUM_ITERATIONS]
                * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_cum_COSTS"].iloc[NUM_ITERATIONS]
                * (1 - FEMALE_PREV)
                + screening_female_trace["dis_cum_COSTS"].iloc[NUM_ITERATIONS]
                * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["cum_COSTS"].iloc[NUM_ITERATIONS]
                * (1 - FEMALE_PREV)
                + screening_female_trace["cum_COSTS"].iloc[NUM_ITERATIONS]
                * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["kf_incidence"].sum() * 100 * (1 - FEMALE_PREV)
                + screening_female_trace["kf_incidence"].sum() * 100 * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_ACE_COSTS"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_ACE_COSTS"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_DAPA_COSTS"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_DAPA_COSTS"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["SCREENING_COSTS"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["SCREENING_COSTS"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_SCREENING_COSTS"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_SCREENING_COSTS"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_DIAGNOSIS_COSTS"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_DIAGNOSIS_COSTS"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(SCREENING_COST + OFFICE_VISIT)
            this_tr_arr.append(
                np.average(
                    screening_male_trace["treated with dapa"],
                    weights=(1 - screening_male_trace["Dead"]),
                )
                * (1 - FEMALE_PREV)
                + np.average(
                    screening_female_trace["treated with dapa"],
                    weights=(1 - screening_female_trace["Dead"]),
                )
                * (FEMALE_PREV)
            )

            this_tr_arr.append(
                screening_male_trace["dis_LY_detected"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_LY_detected"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_QALY_detected"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_QALY_detected"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                screening_male_trace["dis_cost_detected"].sum() * (1 - FEMALE_PREV)
                + screening_female_trace["dis_cost_detected"].sum() * (FEMALE_PREV)
            )
            this_tr_arr.append(
                (
                    placebo_male_trace[detected_or_treated_list].iloc[0].sum()
                    + screening_male_trace["detected_percent"].sum()
                )
                * (1 - FEMALE_PREV)
                + (
                    placebo_female_trace[detected_or_treated_list].iloc[0].sum()
                    + screening_female_trace["detected_percent"].sum()
                )
                * (FEMALE_PREV)
            )
            this_tr_arr.append(placebo_male_trace[STAGES_WITH_ACE].iloc[0].sum())

            trace_results_tracker.append(this_tr_arr)

    if count == 0:
        trace_results_full_df = pd.DataFrame(
            trace_results_tracker,
            columns=[
                "full_index",
                "Starting age",
                "intervention",
                "LY disc",
                "LY",
                "QALY disc",
                "QALY",
                "cost disc",
                "costs",
                "kf_incidence",
                "ace costs",
                "dapa costs",
                "screening costs",
                "discounted screening costs",
                "diagnosis costs",
                "PSA screening + office",
                "avg on dapa",
                "dis_LY_detected",
                "dis_QALY_detected",
                "dis_cost_detected",
                "detected_total",
                "original treated",
            ],
        )
        count = count + 1

    else:
        this_df = pd.DataFrame(
            trace_results_tracker,
            columns=[
                "full_index",
                "Starting age",
                "intervention",
                "LY disc",
                "LY",
                "QALY disc",
                "QALY",
                "cost disc",
                "costs",
                "kf_incidence",
                "ace costs",
                "dapa costs",
                "screening costs",
                "discounted screening costs",
                "diagnosis costs",
                "PSA screening + office",
                "avg on dapa",
                "dis_LY_detected",
                "dis_QALY_detected",
                "dis_cost_detected",
                "detected_total",
                "original treated",
            ],
        )
        trace_results_full_df = pd.concat([trace_results_full_df, this_df])

end = time.time()
print(end - start)
trace_results_full_df.to_excel(save_p, index=False)
