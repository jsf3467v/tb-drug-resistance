import os

from neo4j import GraphDatabase, Query
from who_catalog import WHOCatalog

# SEED DATA
# Names `strains`, `strain_data`, and `mutations` are referenced by the test
# suite and the EDA notebook (which parse this module), so they are preserved.


genes = [
    {'name': 'rpoB', 'locus': 'Rv0667', 'function': 'RNA polymerase beta subunit', 'drug_target': 'rifampin'},
    {'name': 'katG', 'locus': 'Rv1908c', 'function': 'Catalase-peroxidase', 'drug_target': 'isoniazid'},
    {'name': 'inhA', 'locus': 'Rv1484', 'function': 'Enoyl reductase', 'drug_target': 'isoniazid'},
    {'name': 'embB', 'locus': 'Rv3795', 'function': 'Arabinosyltransferase', 'drug_target': 'ethambutol'},
    {'name': 'pncA', 'locus': 'Rv2043c', 'function': 'Pyrazinamidase', 'drug_target': 'pyrazinamide'},
    {'name': 'gyrA', 'locus': 'Rv0006', 'function': 'DNA gyrase subunit A', 'drug_target': 'fluoroquinolones'},
    {'name': 'gyrB', 'locus': 'Rv0005', 'function': 'DNA gyrase subunit B', 'drug_target': 'fluoroquinolones'},
    {'name': 'rpoC', 'locus': 'Rv0668', 'function': 'RNA polymerase beta-prime subunit', 'drug_target': None},
    {'name': 'rrs', 'locus': 'MTB000019', 'function': '16S ribosomal RNA', 'drug_target': 'aminoglycosides'},
    {'name': 'eis', 'locus': 'Rv2416c', 'function': 'Enhanced intracellular survival', 'drug_target': 'kanamycin'},
    {'name': 'Rv0678', 'locus': 'Rv0678', 'function': 'MmpR5 transcriptional regulator', 'drug_target': 'bedaquiline'},
    {'name': 'rplC', 'locus': 'Rv0701', 'function': '50S ribosomal protein L3', 'drug_target': 'linezolid'},
]

drugs = [
    {'name': 'rifampin', 'class': 'first-line', 'abbreviation': 'RIF', 'mechanism': 'RNA polymerase inhibitor'},
    {'name': 'isoniazid', 'class': 'first-line', 'abbreviation': 'INH', 'mechanism': 'Cell wall synthesis inhibitor'},
    {'name': 'ethambutol', 'class': 'first-line', 'abbreviation': 'EMB', 'mechanism': 'Cell wall synthesis inhibitor'},
    {'name': 'pyrazinamide', 'class': 'first-line', 'abbreviation': 'PZA', 'mechanism': 'Disrupts membrane potential'},
    {'name': 'levofloxacin', 'class': 'second-line', 'abbreviation': 'LFX', 'mechanism': 'DNA gyrase inhibitor'},
    {'name': 'moxifloxacin', 'class': 'second-line', 'abbreviation': 'MFX', 'mechanism': 'DNA gyrase inhibitor'},
    {'name': 'amikacin', 'class': 'second-line-injectable', 'abbreviation': 'AMK', 'mechanism': 'Protein synthesis inhibitor'},
    {'name': 'capreomycin', 'class': 'second-line-injectable', 'abbreviation': 'CAP', 'mechanism': 'Protein synthesis inhibitor'},
    {'name': 'kanamycin', 'class': 'second-line-injectable', 'abbreviation': 'KAN', 'mechanism': 'Protein synthesis inhibitor'},
    {'name': 'bedaquiline', 'class': 'new-drug', 'abbreviation': 'BDQ', 'mechanism': 'ATP synthase inhibitor'},
    {'name': 'delamanid', 'class': 'new-drug', 'abbreviation': 'DLM', 'mechanism': 'Mycolic acid synthesis inhibitor'},
    {'name': 'linezolid', 'class': 'repurposed', 'abbreviation': 'LZD', 'mechanism': 'Protein synthesis inhibitor'},
    {'name': 'clofazimine', 'class': 'repurposed', 'abbreviation': 'CFZ', 'mechanism': 'Multiple mechanisms'},
    {'name': 'pretomanid', 'class': 'new-drug', 'abbreviation': 'PA', 'mechanism': 'Cell wall synthesis inhibitor'},
]

resistance_profiles = [
    {'type': 'Susceptible', 'abbreviation': 'S', 'description': 'No resistance detected'},
    {'type': 'MonoResistant', 'abbreviation': 'MR', 'description': 'Resistant to one first-line drug'},
    {'type': 'PolyResistant', 'abbreviation': 'PR', 'description': 'Resistant to >1 first-line drug (not MDR)'},
    {'type': 'MDR', 'abbreviation': 'MDR', 'description': 'Resistant to at least isoniazid and rifampin'},
    {'type': 'PreXDR', 'abbreviation': 'PreXDR', 'description': 'MDR + resistant to fluoroquinolone OR injectable (pre-2021 definition)'},
    {'type': 'XDR', 'abbreviation': 'XDR', 'description': 'MDR + resistant to fluoroquinolone AND injectable (pre-2021 definition)'},
]

mutations = [
    {'id': 'rpoB_p.Ser450Leu', 'gene': 'rpoB', 'position': 450, 'ref': 'S', 'alt': 'L', 'drug': 'rifampin', 'level': 'high'},
    {'id': 'rpoB_p.Asp435Val', 'gene': 'rpoB', 'position': 435, 'ref': 'D', 'alt': 'V', 'drug': 'rifampin', 'level': 'high'},
    {'id': 'rpoB_p.His445Tyr', 'gene': 'rpoB', 'position': 445, 'ref': 'H', 'alt': 'Y', 'drug': 'rifampin', 'level': 'high'},
    {'id': 'rpoB_p.His445Asp', 'gene': 'rpoB', 'position': 445, 'ref': 'H', 'alt': 'D', 'drug': 'rifampin', 'level': 'high'},
    {'id': 'rpoB_p.Leu430Pro', 'gene': 'rpoB', 'position': 430, 'ref': 'L', 'alt': 'P', 'drug': 'rifampin', 'level': 'moderate'},
    {'id': 'katG_p.Ser315Thr', 'gene': 'katG', 'position': 315, 'ref': 'S', 'alt': 'T', 'drug': 'isoniazid', 'level': 'high'},
    {'id': 'katG_p.Ser315Asn', 'gene': 'katG', 'position': 315, 'ref': 'S', 'alt': 'N', 'drug': 'isoniazid', 'level': 'high'},
    {'id': 'katG_p.Ser315Ile', 'gene': 'katG', 'position': 315, 'ref': 'S', 'alt': 'I', 'drug': 'isoniazid', 'level': 'high'},
    {'id': 'inhA_c.-15C>T', 'gene': 'inhA', 'position': -15, 'ref': 'C', 'alt': 'T', 'drug': 'isoniazid', 'level': 'low'},
    {'id': 'inhA_c.-8T>C', 'gene': 'inhA', 'position': -8, 'ref': 'T', 'alt': 'C', 'drug': 'isoniazid', 'level': 'low'},
    {'id': 'embB_p.Met306Val', 'gene': 'embB', 'position': 306, 'ref': 'M', 'alt': 'V', 'drug': 'ethambutol', 'level': 'high'},
    {'id': 'embB_p.Met306Ile', 'gene': 'embB', 'position': 306, 'ref': 'M', 'alt': 'I', 'drug': 'ethambutol', 'level': 'high'},
    {'id': 'pncA_p.His57Asp', 'gene': 'pncA', 'position': 57, 'ref': 'H', 'alt': 'D', 'drug': 'pyrazinamide', 'level': 'high'},
    {'id': 'pncA_p.Trp68Arg', 'gene': 'pncA', 'position': 68, 'ref': 'W', 'alt': 'R', 'drug': 'pyrazinamide', 'level': 'high'},
    {'id': 'gyrA_p.Asp94Gly', 'gene': 'gyrA', 'position': 94, 'ref': 'D', 'alt': 'G', 'drug': 'levofloxacin', 'level': 'high'},
    {'id': 'gyrA_p.Asp94Asn', 'gene': 'gyrA', 'position': 94, 'ref': 'D', 'alt': 'N', 'drug': 'levofloxacin', 'level': 'high'},
    {'id': 'gyrA_p.Ala90Val', 'gene': 'gyrA', 'position': 90, 'ref': 'A', 'alt': 'V', 'drug': 'levofloxacin', 'level': 'high'},
    {'id': 'rrs_n.1401A>G', 'gene': 'rrs', 'position': 1401, 'ref': 'A', 'alt': 'G', 'drug': 'amikacin', 'level': 'high'},
    {'id': 'rrs_n.1402C>T', 'gene': 'rrs', 'position': 1402, 'ref': 'C', 'alt': 'T', 'drug': 'amikacin', 'level': 'high'},
    {'id': 'eis_c.-10G>A', 'gene': 'eis', 'position': -10, 'ref': 'G', 'alt': 'A', 'drug': 'kanamycin', 'level': 'moderate'},
    {'id': 'rpoC_p.Val483Gly', 'gene': 'rpoC', 'position': 483, 'ref': 'V', 'alt': 'G', 'drug': 'rifampin', 'level': 'low'},
    {'id': 'Rv0678_c.137_138insG', 'gene': 'Rv0678', 'position': 137, 'ref': '-', 'alt': 'G', 'drug': 'bedaquiline', 'level': 'high'},
    {'id': 'rplC_p.Cys154Arg', 'gene': 'rplC', 'position': 154, 'ref': 'C', 'alt': 'R', 'drug': 'linezolid', 'level': 'high'},
]

strains = [
    {'id': 'TB001', 'lineage': 'Beijing', 'country': 'China', 'year': 2023},
    {'id': 'TB002', 'lineage': 'Euro-American', 'country': 'USA', 'year': 2023},
    {'id': 'TB003', 'lineage': 'Beijing', 'country': 'South Africa', 'year': 2024},
    {'id': 'TB004', 'lineage': 'Indo-Oceanic', 'country': 'India', 'year': 2024},
    {'id': 'TB005', 'lineage': 'Indo-Oceanic', 'country': 'India', 'year': 2023},
    {'id': 'TB006', 'lineage': 'Beijing', 'country': 'China', 'year': 2023},
    {'id': 'TB007', 'lineage': 'Euro-American', 'country': 'Brazil', 'year': 2024},
    {'id': 'TB008', 'lineage': 'Euro-American', 'country': 'USA', 'year': 2023},
    {'id': 'TB009', 'lineage': 'Beijing', 'country': 'Russia', 'year': 2024},
    {'id': 'TB010', 'lineage': 'East-African-Indian', 'country': 'Tanzania', 'year': 2023},
    {'id': 'TB011', 'lineage': 'Beijing', 'country': 'South Africa', 'year': 2023},
    {'id': 'TB012', 'lineage': 'Beijing', 'country': 'China', 'year': 2024},
    {'id': 'TB013', 'lineage': 'Euro-American', 'country': 'Moldova', 'year': 2023},
    {'id': 'TB014', 'lineage': 'Beijing', 'country': 'Kazakhstan', 'year': 2024},
    {'id': 'TB015', 'lineage': 'Indo-Oceanic', 'country': 'Bangladesh', 'year': 2023},
    {'id': 'TB016', 'lineage': 'Euro-American', 'country': 'Nigeria', 'year': 2023},
    {'id': 'TB017', 'lineage': 'East-African-Indian', 'country': 'Kenya', 'year': 2024},
    {'id': 'TB018', 'lineage': 'Euro-American', 'country': 'Peru', 'year': 2023},
    {'id': 'TB019', 'lineage': 'Beijing', 'country': 'Vietnam', 'year': 2024},
    {'id': 'TB020', 'lineage': 'Indo-Oceanic', 'country': 'Indonesia', 'year': 2024},
    {'id': 'TB021', 'lineage': 'Indo-Oceanic', 'country': 'Philippines', 'year': 2024},
    {'id': 'TB022', 'lineage': 'Indo-Oceanic', 'country': 'Pakistan', 'year': 2023},
    {'id': 'TB023', 'lineage': 'Beijing', 'country': 'Myanmar', 'year': 2024},
    {'id': 'TB024', 'lineage': 'East-African-Indian', 'country': 'Ethiopia', 'year': 2023},
    {'id': 'TB025', 'lineage': 'East-African-Indian', 'country': 'Mozambique', 'year': 2024},
    {'id': 'TB026', 'lineage': 'Euro-American', 'country': 'Ukraine', 'year': 2023},
    {'id': 'TB027', 'lineage': 'Euro-American', 'country': 'Mexico', 'year': 2024},
    {'id': 'TB028', 'lineage': 'Beijing', 'country': 'Thailand', 'year': 2023},
    {'id': 'TB029', 'lineage': 'East-African-Indian', 'country': 'DRC', 'year': 2024},
    {'id': 'TB030', 'lineage': 'East-African-Indian', 'country': 'Zimbabwe', 'year': 2023},
    {'id': 'TB031', 'lineage': 'Indo-Oceanic', 'country': 'Papua New Guinea', 'year': 2024},
    {'id': 'TB032', 'lineage': 'Beijing', 'country': 'North Korea', 'year': 2023},
    {'id': 'TB033', 'lineage': 'Euro-American', 'country': 'South Africa', 'year': 2024},
    {'id': 'TB034', 'lineage': 'Indo-Oceanic', 'country': 'India', 'year': 2023},
    {'id': 'TB035', 'lineage': 'Beijing', 'country': 'China', 'year': 2024},
    {'id': 'TB036', 'lineage': 'Euro-American', 'country': 'Romania', 'year': 2023},
    {'id': 'TB037', 'lineage': 'East-African-Indian', 'country': 'Uganda', 'year': 2024},
    {'id': 'TB038', 'lineage': 'Indo-Oceanic', 'country': 'Cambodia', 'year': 2023},
    {'id': 'TB039', 'lineage': 'Beijing', 'country': 'Mongolia', 'year': 2024},
    {'id': 'TB040', 'lineage': 'Euro-American', 'country': 'Colombia', 'year': 2023},
    {'id': 'TB041', 'lineage': 'East-African-Indian', 'country': 'Somalia', 'year': 2024},
    {'id': 'TB042', 'lineage': 'Beijing', 'country': 'Uzbekistan', 'year': 2023},
    {'id': 'TB043', 'lineage': 'Euro-American', 'country': 'Argentina', 'year': 2024},
    {'id': 'TB044', 'lineage': 'Indo-Oceanic', 'country': 'Nepal', 'year': 2023},
    {'id': 'TB045', 'lineage': 'Beijing', 'country': 'Kyrgyzstan', 'year': 2024},
    {'id': 'TB046', 'lineage': 'Euro-American', 'country': 'Georgia', 'year': 2023},
    {'id': 'TB047', 'lineage': 'East-African-Indian', 'country': 'Angola', 'year': 2024},
    {'id': 'TB048', 'lineage': 'Indo-Oceanic', 'country': 'Afghanistan', 'year': 2023},
    {'id': 'TB049', 'lineage': 'Beijing', 'country': 'Tajikistan', 'year': 2024},
    {'id': 'TB050', 'lineage': 'Euro-American', 'country': 'Azerbaijan', 'year': 2023},
    {'id': 'TB051', 'lineage': 'Euro-American', 'country': 'USA', 'year': 2024},
    {'id': 'TB052', 'lineage': 'Indo-Oceanic', 'country': 'India', 'year': 2024},
    {'id': 'TB053', 'lineage': 'Beijing', 'country': 'China', 'year': 2023},
    {'id': 'TB054', 'lineage': 'East-African-Indian', 'country': 'Kenya', 'year': 2024},
    {'id': 'TB055', 'lineage': 'Euro-American', 'country': 'Brazil', 'year': 2023},
    {'id': 'TB056', 'lineage': 'Indo-Oceanic', 'country': 'Philippines', 'year': 2024},
]

patients = [
    {'id': 'P001', 'age': 45, 'sex': 'M', 'hiv_status': 'positive', 'country': 'South Africa', 'region': 'African', 'diabetes': False, 'previous_treatment': True},
    {'id': 'P002', 'age': 32, 'sex': 'F', 'hiv_status': 'negative', 'country': 'India', 'region': 'SE_Asia', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P003', 'age': 67, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Moldova', 'region': 'European', 'diabetes': True, 'previous_treatment': True},
    {'id': 'P004', 'age': 28, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Peru', 'region': 'Americas', 'diabetes': False, 'previous_treatment': True},
    {'id': 'P005', 'age': 51, 'sex': 'M', 'hiv_status': 'negative', 'country': 'China', 'region': 'W_Pacific', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P006', 'age': 39, 'sex': 'F', 'hiv_status': 'positive', 'country': 'Nigeria', 'region': 'African', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P007', 'age': 55, 'sex': 'M', 'hiv_status': 'positive', 'country': 'South Africa', 'region': 'African', 'diabetes': True, 'previous_treatment': True},
    {'id': 'P008', 'age': 24, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Philippines', 'region': 'W_Pacific', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P009', 'age': 41, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Pakistan', 'region': 'E_Mediterranean', 'diabetes': True, 'previous_treatment': True},
    {'id': 'P010', 'age': 36, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Myanmar', 'region': 'SE_Asia', 'diabetes': False, 'previous_treatment': True},
    {'id': 'P011', 'age': 62, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Ukraine', 'region': 'European', 'diabetes': True, 'previous_treatment': True},
    {'id': 'P012', 'age': 29, 'sex': 'F', 'hiv_status': 'positive', 'country': 'Ethiopia', 'region': 'African', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P013', 'age': 48, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Brazil', 'region': 'Americas', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P014', 'age': 33, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Vietnam', 'region': 'W_Pacific', 'diabetes': False, 'previous_treatment': True},
    {'id': 'P015', 'age': 58, 'sex': 'M', 'hiv_status': 'positive', 'country': 'Russia', 'region': 'European', 'diabetes': False, 'previous_treatment': True},
    {'id': 'P016', 'age': 26, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Indonesia', 'region': 'SE_Asia', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P017', 'age': 44, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Bangladesh', 'region': 'SE_Asia', 'diabetes': True, 'previous_treatment': False},
    {'id': 'P018', 'age': 37, 'sex': 'F', 'hiv_status': 'positive', 'country': 'Kenya', 'region': 'African', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P019', 'age': 52, 'sex': 'M', 'hiv_status': 'negative', 'country': 'Kazakhstan', 'region': 'European', 'diabetes': False, 'previous_treatment': False},
    {'id': 'P020', 'age': 31, 'sex': 'F', 'hiv_status': 'negative', 'country': 'Mexico', 'region': 'Americas', 'diabetes': False, 'previous_treatment': False},
]

patient_infections = [
    {'patient': 'P001', 'strain': 'TB003', 'date': '2024-01-15'},
    {'patient': 'P002', 'strain': 'TB005', 'date': '2023-06-20'},
    {'patient': 'P003', 'strain': 'TB011', 'date': '2023-09-10'},
    {'patient': 'P004', 'strain': 'TB019', 'date': '2024-02-28'},
    {'patient': 'P005', 'strain': 'TB001', 'date': '2023-01-10'},
    {'patient': 'P006', 'strain': 'TB016', 'date': '2023-11-05'},
    {'patient': 'P007', 'strain': 'TB033', 'date': '2024-03-12'},
    {'patient': 'P008', 'strain': 'TB021', 'date': '2024-05-08'},
    {'patient': 'P009', 'strain': 'TB022', 'date': '2023-07-22'},
    {'patient': 'P010', 'strain': 'TB023', 'date': '2024-02-18'},
    {'patient': 'P011', 'strain': 'TB026', 'date': '2023-10-30'},
    {'patient': 'P012', 'strain': 'TB024', 'date': '2023-12-05'},
    {'patient': 'P013', 'strain': 'TB007', 'date': '2024-04-14'},
    {'patient': 'P014', 'strain': 'TB019', 'date': '2024-01-25'},
    {'patient': 'P015', 'strain': 'TB009', 'date': '2024-06-11'},
    {'patient': 'P016', 'strain': 'TB020', 'date': '2024-03-07'},
    {'patient': 'P017', 'strain': 'TB015', 'date': '2023-08-19'},
    {'patient': 'P018', 'strain': 'TB017', 'date': '2024-02-22'},
    {'patient': 'P019', 'strain': 'TB014', 'date': '2024-05-30'},
    {'patient': 'P020', 'strain': 'TB027', 'date': '2024-04-03'},
]

strain_data = [
    {'strain': 'TB001', 'mutations': ['rpoB_p.Ser450Leu', 'katG_p.Ser315Thr'], 'profile': 'MDR'},
    {'strain': 'TB002', 'mutations': ['katG_p.Ser315Asn'], 'profile': 'MonoResistant'},
    {'strain': 'TB003', 'mutations': ['rpoB_p.Ser450Leu', 'katG_p.Ser315Thr', 'gyrA_p.Asp94Gly'], 'profile': 'PreXDR'},
    {'strain': 'TB004', 'mutations': ['inhA_c.-15C>T', 'embB_p.Met306Val'], 'profile': 'PolyResistant'},
    {'strain': 'TB005', 'mutations': ['katG_p.Ser315Thr', 'inhA_c.-15C>T'], 'profile': 'MonoResistant'},
    {'strain': 'TB006', 'mutations': ['rpoB_p.Asp435Val', 'katG_p.Ser315Asn'], 'profile': 'MDR'},
    {'strain': 'TB007', 'mutations': ['embB_p.Met306Ile'], 'profile': 'MonoResistant'},
    {'strain': 'TB008', 'mutations': ['katG_p.Ser315Ile'], 'profile': 'MonoResistant'},
    {'strain': 'TB009', 'mutations': ['rpoB_p.His445Tyr', 'katG_p.Ser315Thr', 'embB_p.Met306Val'], 'profile': 'MDR'},
    {'strain': 'TB010', 'mutations': ['inhA_c.-8T>C'], 'profile': 'MonoResistant'},
    {'strain': 'TB011', 'mutations': ['rpoB_p.Ser450Leu', 'katG_p.Ser315Thr', 'gyrA_p.Asp94Gly', 'rrs_n.1401A>G'], 'profile': 'XDR'},
    {'strain': 'TB012', 'mutations': ['rpoB_p.Ser450Leu', 'rpoC_p.Val483Gly', 'katG_p.Ser315Thr'], 'profile': 'MDR'},
    {'strain': 'TB013', 'mutations': ['rpoB_p.His445Asp', 'katG_p.Ser315Asn', 'gyrA_p.Asp94Asn'], 'profile': 'PreXDR'},
    {'strain': 'TB014', 'mutations': ['rpoB_p.Leu430Pro', 'katG_p.Ser315Ile', 'pncA_p.His57Asp'], 'profile': 'MDR'},
    {'strain': 'TB015', 'mutations': ['rpoB_p.Asp435Val', 'inhA_c.-15C>T', 'embB_p.Met306Val', 'gyrA_p.Asp94Gly'], 'profile': 'PreXDR'},
    {'strain': 'TB016', 'mutations': ['katG_p.Ser315Thr'], 'profile': 'MonoResistant'},
    {'strain': 'TB017', 'mutations': ['inhA_c.-15C>T'], 'profile': 'MonoResistant'},
    {'strain': 'TB018', 'mutations': ['rpoB_p.Ser450Leu'], 'profile': 'MonoResistant'},
    {'strain': 'TB019', 'mutations': ['rpoB_p.His445Tyr', 'katG_p.Ser315Thr', 'pncA_p.Trp68Arg'], 'profile': 'MDR'},
    {'strain': 'TB020', 'mutations': ['katG_p.Ser315Asn', 'embB_p.Met306Ile'], 'profile': 'PolyResistant'},
    {'strain': 'TB021', 'mutations': ['embB_p.Met306Val'], 'profile': 'MonoResistant'},
    {'strain': 'TB022', 'mutations': ['katG_p.Ser315Thr', 'pncA_p.His57Asp'], 'profile': 'PolyResistant'},
    {'strain': 'TB023', 'mutations': ['rpoB_p.His445Tyr', 'katG_p.Ser315Asn'], 'profile': 'MDR'},
    {'strain': 'TB024', 'mutations': ['inhA_c.-15C>T'], 'profile': 'MonoResistant'},
    {'strain': 'TB025', 'mutations': ['katG_p.Ser315Ile', 'embB_p.Met306Ile'], 'profile': 'PolyResistant'},
    {'strain': 'TB026', 'mutations': ['rpoB_p.Ser450Leu', 'katG_p.Ser315Thr', 'gyrA_p.Ala90Val'], 'profile': 'PreXDR'},
    {'strain': 'TB027', 'mutations': ['pncA_p.Trp68Arg'], 'profile': 'MonoResistant'},
    {'strain': 'TB028', 'mutations': ['katG_p.Ser315Asn'], 'profile': 'MonoResistant'},
    {'strain': 'TB029', 'mutations': ['embB_p.Met306Val', 'pncA_p.His57Asp'], 'profile': 'PolyResistant'},
    {'strain': 'TB030', 'mutations': ['inhA_c.-8T>C'], 'profile': 'MonoResistant'},
    {'strain': 'TB031', 'mutations': ['katG_p.Ser315Thr'], 'profile': 'MonoResistant'},
    {'strain': 'TB032', 'mutations': ['rpoB_p.Asp435Val', 'katG_p.Ser315Ile'], 'profile': 'MDR'},
    {'strain': 'TB033', 'mutations': ['rpoB_p.His445Asp', 'katG_p.Ser315Thr'], 'profile': 'MDR'},
    {'strain': 'TB034', 'mutations': ['inhA_c.-15C>T', 'pncA_p.Trp68Arg'], 'profile': 'PolyResistant'},
    {'strain': 'TB035', 'mutations': ['rpoB_p.Ser450Leu', 'katG_p.Ser315Asn', 'gyrA_p.Asp94Gly', 'rrs_n.1402C>T'], 'profile': 'XDR'},
    {'strain': 'TB036', 'mutations': ['embB_p.Met306Ile'], 'profile': 'MonoResistant'},
    {'strain': 'TB037', 'mutations': ['katG_p.Ser315Thr', 'embB_p.Met306Val'], 'profile': 'PolyResistant'},
    {'strain': 'TB038', 'mutations': ['inhA_c.-15C>T'], 'profile': 'MonoResistant'},
    {'strain': 'TB039', 'mutations': ['rpoB_p.Leu430Pro', 'katG_p.Ser315Thr'], 'profile': 'MDR'},
    {'strain': 'TB040', 'mutations': ['pncA_p.His57Asp'], 'profile': 'MonoResistant'},
    {'strain': 'TB041', 'mutations': ['katG_p.Ser315Asn', 'embB_p.Met306Ile', 'pncA_p.Trp68Arg'], 'profile': 'PolyResistant'},
    {'strain': 'TB042', 'mutations': ['rpoB_p.His445Tyr', 'katG_p.Ser315Ile', 'gyrA_p.Asp94Asn'], 'profile': 'PreXDR'},
    {'strain': 'TB043', 'mutations': ['embB_p.Met306Val'], 'profile': 'MonoResistant'},
    {'strain': 'TB044', 'mutations': ['katG_p.Ser315Thr'], 'profile': 'MonoResistant'},
    {'strain': 'TB045', 'mutations': ['rpoB_p.Asp435Val', 'katG_p.Ser315Asn', 'gyrA_p.Ala90Val'], 'profile': 'PreXDR'},
    {'strain': 'TB046', 'mutations': ['inhA_c.-8T>C'], 'profile': 'MonoResistant'},
    {'strain': 'TB047', 'mutations': ['katG_p.Ser315Ile', 'pncA_p.His57Asp'], 'profile': 'PolyResistant'},
    {'strain': 'TB048', 'mutations': ['rpoB_p.Ser450Leu'], 'profile': 'MonoResistant'},
    {'strain': 'TB049', 'mutations': ['rpoB_p.His445Asp', 'katG_p.Ser315Thr', 'gyrA_p.Asp94Gly'], 'profile': 'PreXDR'},
    {'strain': 'TB050', 'mutations': ['katG_p.Ser315Asn'], 'profile': 'MonoResistant'},
    {'strain': 'TB051', 'mutations': [], 'profile': 'Susceptible'},
    {'strain': 'TB052', 'mutations': [], 'profile': 'Susceptible'},
    {'strain': 'TB053', 'mutations': [], 'profile': 'Susceptible'},
    {'strain': 'TB054', 'mutations': [], 'profile': 'Susceptible'},
    {'strain': 'TB055', 'mutations': [], 'profile': 'Susceptible'},
    {'strain': 'TB056', 'mutations': [], 'profile': 'Susceptible'},
]

transmissions = [
    {'source': 'TB001', 'target': 'TB003', 'location': 'household', 'date': '2023-12-01'},
    {'source': 'TB005', 'target': 'TB006', 'location': 'workplace', 'date': '2023-08-15'},
    {'source': 'TB011', 'target': 'TB012', 'location': 'prison', 'date': '2023-11-20'},
    {'source': 'TB023', 'target': 'TB028', 'location': 'household', 'date': '2024-01-10'},
    {'source': 'TB026', 'target': 'TB046', 'location': 'healthcare', 'date': '2023-11-05'},
    {'source': 'TB032', 'target': 'TB035', 'location': 'household', 'date': '2024-03-22'},
    {'source': 'TB042', 'target': 'TB045', 'location': 'prison', 'date': '2024-02-14'},
]


# ONTOLOGY

# Node keys that get a lookup index and a uniqueness constraint. Held as data so
# the DDL can be rendered for either the Memgraph or the Neo4j dialect.
KEY_SPECS = (
    ('Gene', 'name'),
    ('Drug', 'name'),
    ('Strain', 'strain_id'),
    ('Mutation', 'mutation_id'),
    ('Patient', 'patient_id'),
)


class TBOntology:
    """TB Mutations Ontology Class"""

    def __init__(self, uri=None, user=None, password=None):
        uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = user or os.getenv('NEO4J_USER')
        password = password or os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password) if user else None,
            connection_timeout=10.0,
            max_transaction_retry_time=10.0
        )

    def _batch(self, query, rows):
        """Run one UNWIND query over a list of row dicts in a single round trip."""
        with self.driver.session() as session:
            session.run(query, {'rows': rows})

    def clear_database(self):
        """Clear all data from database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def schema(self):
        """Lookup index and uniqueness constraint for each node key, in whichever
        dialect the backend accepts. Statements neither dialect applies are
        reported, not silently ignored."""
        failed = []
        for label, prop in KEY_SPECS:
            pairs = [
                (f"CREATE INDEX ON :{label}({prop})",
                 f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"),
                (f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{prop} IS UNIQUE",
                 f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"),
            ]
            for primary, fallback in pairs:
                note = self._statement(primary, fallback)
                if note:
                    failed.append(note)
        if failed:
            print(f"Schema: {len(failed)} statement(s) not applied")
            for note in failed:
                print(f"  {note}")

    def _statement(self, primary, fallback):
        """Apply a DDL statement, trying the fallback dialect on a syntax error and
        treating an already-created result as success. Returns a short note if
        neither dialect applied, else None."""
        last = None
        for statement in (primary, fallback):
            try:
                with self.driver.session() as session:
                    session.run(statement)
                return None
            except Exception as exc:
                if 'already exists' in str(exc).lower():
                    return None
                last = exc
        return f"{primary} -> {last}"

    def ontology_classes(self):
        """All nodes and relationships"""
        self._genes()
        self._drugs()
        self._resistance_profiles()
        self._mutations()
        self._strains()
        self._patients()
        self._relationships()
        self._transmission()

    def _genes(self):
        """Gene nodes"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (g:Gene {name: row.name})
            SET g.locus = row.locus, g.function = row.function, g.drug_target = row.drug_target
        """, genes)

    def _drugs(self):
        """Drug nodes"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (d:Drug {name: row.name})
            SET d.class = row.class, d.abbreviation = row.abbreviation, d.mechanism = row.mechanism
        """, drugs)

    def _resistance_profiles(self):
        """Resistance profile nodes"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (r:ResistanceProfile {type: row.type})
            SET r.abbreviation = row.abbreviation, r.description = row.description
        """, resistance_profiles)

    def _mutations(self):
        """Mutation nodes and links to genes and drugs"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (m:Mutation {mutation_id: row.id})
            SET m.position = row.position, m.ref_amino_acid = row.ref, m.alt_amino_acid = row.alt
        """, mutations)
        self._batch("""
            UNWIND $rows AS row
            MATCH (m:Mutation {mutation_id: row.id}), (g:Gene {name: row.gene})
            MERGE (m)-[:IN_GENE]->(g)
        """, mutations)
        self._batch("""
            UNWIND $rows AS row
            MATCH (m:Mutation {mutation_id: row.id}), (d:Drug {name: row.drug})
            MERGE (m)-[rel:CONFERS_RESISTANCE]->(d)
            SET rel.level = row.level
        """, mutations)

    def _strains(self):
        """Strain nodes"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (s:Strain {strain_id: row.id})
            SET s.lineage = row.lineage, s.country = row.country, s.year = row.year
        """, strains)

    def _patients(self):
        """Patient nodes and their infections"""
        self._batch("""
            UNWIND $rows AS row
            MERGE (p:Patient {patient_id: row.id})
            SET p.age = row.age, p.sex = row.sex, p.hiv_status = row.hiv_status,
                p.country = row.country, p.region = row.region,
                p.diabetes = row.diabetes, p.previous_treatment = row.previous_treatment
        """, patients)
        self._batch("""
            UNWIND $rows AS row
            MATCH (p:Patient {patient_id: row.patient}), (s:Strain {strain_id: row.strain})
            MERGE (p)-[:INFECTED_WITH {date: date(row.date)}]->(s)
        """, patient_infections)

    def _relationships(self):
        """Relationships between strains, mutations, and profiles"""
        pairs = [{'strain': r['strain'], 'mutation': m}
                 for r in strain_data for m in r['mutations']]
        strain_profiles = [{'strain': r['strain'], 'profile': r['profile']}
                           for r in strain_data]
        self._batch("""
            UNWIND $rows AS row
            MATCH (s:Strain {strain_id: row.strain}), (m:Mutation {mutation_id: row.mutation})
            MERGE (s)-[:HAS_MUTATION]->(m)
        """, pairs)
        self._batch("""
            UNWIND $rows AS row
            MATCH (s:Strain {strain_id: row.strain}), (r:ResistanceProfile {type: row.profile})
            MERGE (s)-[:HAS_PROFILE]->(r)
        """, strain_profiles)

    def _transmission(self):
        """Transmission relationships between strains"""
        self._batch("""
            UNWIND $rows AS row
            MATCH (s1:Strain {strain_id: row.source}), (s2:Strain {strain_id: row.target})
            MERGE (s1)-[:TRANSMITTED_TO {location: row.location, date: date(row.date)}]->(s2)
        """, transmissions)

    def query(self, cypher_query, parameters=None):
        """Execute any Cypher query and return results"""
        with self.driver.session() as session:
            q = Query(cypher_query, timeout=30.0)
            result = session.run(q, parameters or {})
            return [record.data() for record in result]

    def read_query(self, cypher_query, parameters=None):
        """Execute a read-only Cypher query; the database rejects any write."""
        with self.driver.session() as session:
            return session.execute_read(
                lambda tx: [record.data() for record in tx.run(cypher_query, parameters or {})]
            )

    def who_mutations(self, filepath=None):
        catalog = WHOCatalog(filepath)

        total = 0
        with self.driver.session() as session:
            for batch in catalog.batch_mutations(batch_size=1000):
                session.run("""
                    UNWIND $mutations AS mut
                    MERGE (g:Gene {name: mut.gene})
                    MERGE (d:Drug {name: mut.drug})
                    MERGE (m:Mutation {mutation_id: mut.mutation_id})
                    SET m.confidence = mut.confidence,
                        m.tier = mut.tier
                    MERGE (m)-[:IN_GENE]->(g)
                    MERGE (m)-[rel:CONFERS_RESISTANCE]->(d)
                    SET rel.level = mut.confidence
                """, {'mutations': batch})
                total += len(batch)

        print(f"Merged {total:,} WHO mutations")

    def count_who_mutations(self):
        """Count mutation nodes carrying WHO catalog data (a confidence tier).
        WHO rows share mutation_ids across drugs, so this node count is lower than
        the number of merged WHO rows."""
        query = """
            MATCH (m:Mutation)
            WHERE m.confidence IS NOT NULL
            RETURN count(m) as total
        """

        with self.driver.session() as session:
            result = session.run(query)
            total = result.single()['total'] if result else 0
        print(f"WHO-sourced mutation nodes: {total}")
        return total

    def count_nodes(self):
        """Count of all nodes by type"""
        query = """
            MATCH (n)
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY type
        """
        return self.query(query)

    def strain_mutations_detailed(self, strain_id):
        """Detailed mutation info for a strain"""
        query = """
            MATCH (s:Strain {strain_id: $strain_id})-[:HAS_MUTATION]->(m:Mutation)
            OPTIONAL MATCH (m)-[:IN_GENE]->(g:Gene)
            OPTIONAL MATCH (m)-[:CONFERS_RESISTANCE]->(d:Drug)
            RETURN m.mutation_id as mutation, g.name as gene,
                   m.position as position, d.name as drug,
                   m.confidence as confidence
            ORDER BY g.name, m.position
        """
        return self.query(query, {'strain_id': strain_id})

    def patient_strain_mapping(self, patient_id):
        """Map patient to their strain and profile"""
        query = """
            MATCH (p:Patient {patient_id: $patient_id})-[:INFECTED_WITH]->(s:Strain)
            OPTIONAL MATCH (s)-[:HAS_PROFILE]->(r:ResistanceProfile)
            RETURN s.strain_id as strain, r.type as profile,
                   s.lineage as lineage, s.country as country
        """
        return self.query(query, {'patient_id': patient_id})

    def close(self):
        """Close database connection"""
        self.driver.close()


def main():
    """Run all"""
    ontology = TBOntology()
    ontology.clear_database()
    ontology.schema()
    ontology.ontology_classes()

    try:
        ontology.who_mutations()
        ontology.count_who_mutations()
    except Exception as e:
        print(f"WHO data step skipped: {str(e)}")

    print("Database initialized successfully")
    ontology.close()


if __name__ == "__main__":
    main()