import argparse
import pandas as pd
import random
import datetime
import numpy as np
import os

# -------------------------------------------------------------------------
# CONSTANTS & DICTIONARIES
# -------------------------------------------------------------------------

CONSTANT_FEATURES = {
    "birthyear": (1945, 1975),
    "gender": ["Male", "Female"],
    "histology": ["Adenocarcinoma", "Squamous Cell Carcinoma"],
    "smoking_history": ["Never", "Former", "Current"],
}

CONSTANT_FEATURES_DESCRIPTION = {
    "patientid": "Unique patient identifier",
    "birthyear": "Age of patient at first event",
    "gender": "Gender of the patient",
    "histology": "Histological subtype of NSCLC",
    "smoking_history": "Smoking status at diagnosis",
}

# NSCLC Specific Lab Tests (loinc_code: (name, unit, mean, std_dev))
LAB_TESTS = {
    "718-7": ("hemoglobin", "g/dL", 13.5, 1.5),
    "26464-8": ("leukocytes", "10*9/L", 7.0, 2.0),
    "26515-7": ("platelets", "10*9/L", 250, 60),
    "2160-0": ("creatinine", "mg/dL", 0.9, 0.3),
    "1742-6": ("alanine aminotransferase", "U/L", 25, 10),
    "1751-7": ("albumin", "g/L", 40, 5),
    "48642-3": ("eGFR", "ml/min/1.73m2", 80, 20),
    "2951-2": ("sodium", "mmol/L", 140, 3),
    "2823-3": ("potassium", "mmol/L", 4.0, 0.4),
}

# NSCLC Regimens
REGIMENS = {
    "chemo_io": ["Carboplatin", "Pemetrexed", "Pembrolizumab"],
    "targeted_egfr": ["Osimertinib"],
    "targeted_alk": ["Alectinib"],
    "second_line_chemo": ["Docetaxel", "Ramucirumab"],
    "second_line_io": ["Nivolumab"],
}

DIAGNOSIS_CODES = [
    ("C34.1", "Malignant neoplasm of upper lobe, bronchus or lung"),
    ("C34.3", "Malignant neoplasm of lower lobe, bronchus or lung"),
    ("C34.90", "Malignant neoplasm of unspecified part of unspecified bronchus or lung"),
]

# Standard NGS Panel for Lung Cancer
GENE_PANEL = ["EGFR", "ALK", "ROS1", "BRAF", "KRAS", "MET", "RET", "NTRK1", "NTRK2", "NTRK3", "HER2"]
# Common co-mutations to add "noise"
CO_MUTATIONS = ["TP53", "STK11", "KEAP1"]

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------


def get_random_date(start_year, end_year):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    return start + datetime.timedelta(days=random.randint(0, (end - start).days))


def add_days(date_obj, days):
    return date_obj + datetime.timedelta(days=days)


def generate_lab_event(patientid, current_date, start_date, lab_code, lab_trends):
    """
    Generates a lab value based on a linear trend specific to the patient.
    Formula: Value = Baseline + (Slope * Days_Elapsed) + Noise
    """
    name, unit, mean, std = LAB_TESTS[lab_code]

    # Retrieve patient-specific trend parameters
    patient_params = lab_trends[lab_code]
    baseline = patient_params["baseline"]
    slope = patient_params["slope"]

    # Calculate days elapsed since diagnosis (or start date)
    days_elapsed = (current_date - start_date).days

    # Calculate linear trend value
    trend_value = baseline + (slope * days_elapsed)

    # Add slight random noise (intra-patient variation is usually smaller than population std)
    # We use std / 10 to ensure the trend remains visible over the noise.
    noise = np.random.normal(0, std / 10.0)

    val = trend_value + noise
    val = max(0, round(val, 2))  # Ensure positive

    clean_name = f"{name} - {lab_code}"
    desc = f"Test: {name}, Cleaned lab units: {unit}"

    return [patientid, current_date, "lab", clean_name.replace(" ", "_"), str(val), clean_name, desc]


def generate_vital_event(patientid, date, vital_type):
    if vital_type == "body_weight":
        val = np.random.normal(75, 15)
        return [patientid, date, "vitals", "body_weight", str(round(val, 2)), "body weight", "NA"]
    elif vital_type == "ecog":
        val = random.choice(["0", "1", "2"])
        return [patientid, date, "ecog", "ecog", val, "ECOG", "NA"]
    return None


def generate_genomic_profile(patientid, test_date, mutation_status):
    """
    Generates a list of genetic results based on the patient's hidden driver status.
    Output aligns with columns:
    [patientid, date, biomarker_category, biomarker_event, biomarker_descriptive_name, biomarker_value, meta_data]
    """
    genetic_records = []

    # 1. Generate Panel Results (Primary Driver + Negatives)
    for gene in GENE_PANEL:
        # Default to negative/wild type
        is_positive = False
        variant = "Wild Type"

        # Check if this gene is the driver
        if gene == mutation_status:
            is_positive = True
            if gene == "EGFR":
                variant = random.choice(["Exon 19 Deletion", "L858R", "Exon 20 Insertion"])
            elif gene == "ALK":
                variant = "EML4-ALK Fusion"
            elif gene == "KRAS":
                variant = "G12C"
            else:
                variant = "Gain of Function"

        # Check for sporadic KRAS in WT patients
        elif gene == "KRAS" and mutation_status == "WT":
            if random.random() < 0.25:
                is_positive = True
                variant = random.choice(["G12C", "G12D", "G12V"])

        # --- Construct Descriptive Name & Value ---
        if is_positive:
            # Descriptive: "EGFR L858R", Value: "Present"
            descriptive_name = f"{gene} {variant}"
            biomarker_value = "Present"
        else:
            # Descriptive: "EGFR", Value: "Wild Type"
            descriptive_name = gene
            biomarker_value = "Wild Type"

        genetic_records.append(
            [
                patientid,
                test_date,
                "basic_biomarker",  # biomarker_category
                gene,  # biomarker_event
                descriptive_name,  # biomarker_descriptive_name
                biomarker_value,  # biomarker_value
                "NGS",  # meta_data
            ]
        )

    # 2. Generate Co-mutations (Noise)
    for gene in CO_MUTATIONS:
        # 40% chance of having a TP53 or STK11 mutation
        if random.random() < 0.4:
            variant = "Missense"
            descriptive_name = f"{gene} {variant}"

            genetic_records.append(
                [
                    patientid,
                    test_date,
                    "gene_sv",  # biomarker_category
                    gene,  # biomarker_event
                    descriptive_name,  # biomarker_descriptive_name
                    "Present",  # biomarker_value
                    "NGS",  # meta_data
                ]
            )

    # 3. Generate PD-L1 (IHC)
    pdl1_score = random.choice(["< 1%", "1-49%", ">= 50%"])

    # Descriptive: "PD-L1 Expression", Value: the actual score range
    genetic_records.append(
        [
            patientid,
            test_date,
            "biomarker_ihc",  # biomarker_category
            "PD-L1",  # biomarker_event
            "PD-L1 Expression (TPS)",  # biomarker_descriptive_name
            pdl1_score,  # biomarker_value
            "IHC 22C3",  # meta_data
        ]
    )

    # 4. Generate Signatures (TMB)
    if mutation_status in ["EGFR", "ALK"]:
        tmb_val = round(np.random.normal(5, 2), 2)
    else:
        tmb_val = round(np.random.normal(12, 5), 2)

    genetic_records.append(
        [
            patientid,
            test_date,
            "signature",  # biomarker_category
            "TMB",  # biomarker_event
            "Tumor Mutational Burden",  # biomarker_descriptive_name
            max(0, tmb_val),  # biomarker_value
            "NGS",  # meta_data
        ]
    )

    return genetic_records


def main(num_patients_to_generate, seed, save_folder):
    random.seed(seed)
    np.random.seed(seed)

    all_constant = []
    all_events = []
    all_genetic = []

    event_columns = [
        "patientid",
        "date",
        "event_category",
        "event_name",
        "event_value",
        "event_descriptive_name",
        "meta_data",
    ]

    genetic_columns = [
        "patientid",
        "date",
        "event_category",  # basic_biomarker, gene_sv, signature
        "event_name",  # EGFR, ALK, TMB
        "event_descriptive_name",  # e.g. "EGFR L858R", "PD-L1 Expression"
        "event_value",  # "Present", "Wild Type", ">= 50%"
        "meta_data",
    ]

    print(f"Generating {num_patients_to_generate} patients with NSCLC trajectories...")

    #: iterate over number of patients
    for patient_idx in range(num_patients_to_generate):
        patientid = f"p{patient_idx}"

        # 1. Generate Constant Data
        constant_data = {"patientid": patientid}
        for const_feat, const_val in CONSTANT_FEATURES.items():
            if isinstance(const_val, list):
                chosen_val = random.choice(const_val)
                constant_data[const_feat] = chosen_val
            else:
                chosen_val = random.randint(const_val[0], const_val[1])
                constant_data[const_feat] = chosen_val

        all_constant.append(constant_data)

        # -----------------------------------------------------------------
        # PRE-CALCULATE LAB TRENDS FOR THIS PATIENT
        # -----------------------------------------------------------------
        patient_lab_trends = {}
        for lab_code, (name, unit, mean, std) in LAB_TESTS.items():
            # 1. Establish a baseline for this patient (offset from population mean)
            baseline = np.random.normal(mean, std)

            # 2. Determine a slope (change per day).
            # We assume a max drift of roughly 1.5 standard deviations over a year (365 days)
            # This ensures the trend is visible but not immediately catastrophic.
            max_daily_drift = (std * 1.5) / 365.0
            slope = random.uniform(-max_daily_drift, max_daily_drift)

            patient_lab_trends[lab_code] = {"baseline": baseline, "slope": slope}
        # -----------------------------------------------------------------

        # 2. Determine Genetic Profile (Hidden driver of the simulation)
        # 15% EGFR+, 5% ALK+, rest Wild Type
        r = random.random()
        if r < 0.15:
            mutation_status = "EGFR"
        elif r < 0.20:
            mutation_status = "ALK"
        else:
            mutation_status = "WT"

        # 3. Generate Events Data

        # A. Diagnosis
        diagnosis_date = get_random_date(2015, 2022)
        dx_code, dx_desc = random.choice(DIAGNOSIS_CODES)

        all_events.append(
            [
                patientid,
                diagnosis_date,
                "main_diagnosis",
                "initial_diagnosis",
                "NSCLC",
                "initial cancer diagnosis",
                "Non-Small Cell Lung Cancer",
            ]
        )
        all_events.append(
            [patientid, diagnosis_date, "diagnosis", dx_code, "diagnosed", f"{dx_desc} ({dx_code})", dx_desc]
        )

        # B. Biomarker Testing (Genetic Data in Event Stream)
        test_date = add_days(diagnosis_date, random.randint(7, 21))

        # Add a high level event indicating testing happened
        all_events.append(
            [patientid, test_date, "lab", "biomarker_test", "performed", "Molecular Profiling", "NGS Panel"]
        )

        # --- GENERATE GENETIC DATA HERE ---
        patient_genetics = generate_genomic_profile(patientid, test_date, mutation_status)
        all_genetic.extend(patient_genetics)
        # ----------------------------------

        # Add clinical summaries of genetics to the event stream for easy reading
        if mutation_status == "EGFR":
            all_events.append(
                [
                    patientid,
                    test_date,
                    "biomarker",
                    "EGFR",
                    "Positive",
                    "Epidermal Growth Factor Receptor",
                    "Exon 19/21",
                ]
            )
        elif mutation_status == "ALK":
            all_events.append(
                [patientid, test_date, "biomarker", "ALK", "Positive", "Anaplastic Lymphoma Kinase", "Translocation"]
            )
        else:
            all_events.append(
                [patientid, test_date, "biomarker", "EGFR", "Negative", "Epidermal Growth Factor Receptor", "WT"]
            )
            all_events.append(
                [patientid, test_date, "biomarker", "ALK", "Negative", "Anaplastic Lymphoma Kinase", "WT"]
            )

        # C. Treatment Trajectory
        current_date = add_days(test_date, random.randint(7, 14))

        # Determine number of lines (1 to 3) based on survival/progression logic
        num_lines = random.choice([1, 1, 2, 2, 3])

        for line_num in range(1, num_lines + 1):
            # Select Regimen based on Line and Genetics
            regimen_drugs = []
            regimen_name = ""

            if line_num == 1:
                if mutation_status == "EGFR":
                    regimen_drugs = REGIMENS["targeted_egfr"]
                    regimen_name = "Osimertinib Monotherapy"
                elif mutation_status == "ALK":
                    regimen_drugs = REGIMENS["targeted_alk"]
                    regimen_name = "Alectinib Monotherapy"
                else:
                    regimen_drugs = REGIMENS["chemo_io"]
                    regimen_name = "Carboplatin/Pemetrexed/Pembrolizumab"
            elif line_num == 2:
                if mutation_status in ["EGFR", "ALK"]:
                    regimen_drugs = REGIMENS["chemo_io"]  # Switch to chemo after TKI fails
                    regimen_name = "Platinum Doublet"
                else:
                    regimen_drugs = REGIMENS["second_line_chemo"]
                    regimen_name = "Docetaxel/Ramucirumab"
            else:
                regimen_drugs = REGIMENS["second_line_io"]  # Salvage
                regimen_name = "Nivolumab"

            line_start_date = current_date

            # Define length of this line (e.g., 3 to 12 months)
            line_duration_days = random.randint(90, 360)
            line_end_date = add_days(line_start_date, line_duration_days)

            # Record Line Start Metadata
            regimen_str = ",".join(regimen_drugs)
            all_events.append([patientid, line_start_date, "lot", "line_number", str(line_num), "line number", "NA"])
            all_events.append([patientid, line_start_date, "lot", "line_name", regimen_name, "line of therapy", "NA"])
            all_events.append(
                [patientid, line_start_date, "lot", "is_maintenance_therapy", "FALSE", "is maintenance therapy", "NA"]
            )

            for drug in regimen_drugs:
                all_events.append([patientid, line_start_date, "lot", drug.lower(), "LoT Start", "LoT", "NA"])

            # Simulate Cycles/Visits within this line
            # Cycles are typically every 21 or 28 days, with some noise added
            cycle_length = 21 + random.choice([0, 0, 0, 0, 0, 7, 7])  # Mostly 21 days
            visit_date = line_start_date

            while visit_date < line_end_date:
                # 1. Labs (Complete Blood Count & Chemistry)
                for lab_code in LAB_TESTS.keys():
                    # UPDATED CALL: Pass diagnosis_date and patient_lab_trends
                    all_events.append(
                        generate_lab_event(patientid, visit_date, diagnosis_date, lab_code, patient_lab_trends)
                    )

                # 2. Vitals
                all_events.append(generate_vital_event(patientid, visit_date, "body_weight"))
                if random.random() < 0.3:  # ECOG recorded less frequently
                    all_events.append(generate_vital_event(patientid, visit_date, "ecog"))

                # 3. Drug Administration
                for drug in regimen_drugs:
                    all_events.append(
                        [patientid, visit_date, "drug", drug.lower(), "administered", drug.lower(), regimen_str]
                    )

                # Advance time
                visit_date = add_days(visit_date, cycle_length)

            # Gap before next line (progression wash-out)
            current_date = add_days(line_end_date, random.randint(14, 45))

            # Add a progression event, radomly since sometimes patients stop for toxicity
            if random.random() < 0.6:
                all_events.append(
                    [
                        patientid,
                        line_end_date,
                        "progression",
                        "progression",
                        "progression",
                        "progression",
                        "Radiological PD",
                    ]
                )

        # Add death event at end of trajectory
        death_date = add_days(current_date, random.randint(30, 180))
        all_events.append([patientid, death_date, "death", "death", "death", "Death", "Cause: Cancer Progression"])

    # Generate constant_description dataframe
    df_constant_description = pd.DataFrame(CONSTANT_FEATURES_DESCRIPTION.items(), columns=["variable", "comment"])

    # Convert all to dataframes
    df_constant = pd.DataFrame(all_constant)
    df_events = pd.DataFrame(all_events, columns=event_columns)
    df_genetic = pd.DataFrame(all_genetic, columns=genetic_columns)

    # Merge genetic data into events
    df_events["source"] = "events"
    df_genetic["source"] = "genetic"
    df_events = pd.concat([df_events, df_genetic], ignore_index=True)

    # Sort events by patient and date
    df_events.sort_values(by=["patientid", "date"], inplace=True)

    # Create Output Directory
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save Data
    df_constant.to_csv(os.path.join(save_folder, "constant.csv"), index=False)
    df_events.to_csv(os.path.join(save_folder, "events.csv"), index=False)
    df_constant_description.to_csv(os.path.join(save_folder, "constant_description.csv"), index=False)

    print(f"Successfully generated data for {num_patients_to_generate} patients in '{save_folder}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example patient data.")
    parser.add_argument(
        "--num_patients_to_generate",
        type=int,
        default=50,
        help="Number of patients to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Folder to save the generated example data.",
    )

    args = parser.parse_args()

    main(args.num_patients_to_generate, args.seed, args.save_folder)
