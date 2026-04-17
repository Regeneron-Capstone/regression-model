"""
Stage 2 — ICD-10 Lookup via local TF-IDF search against DXCCSR

Input:  2_condition_mapping/output/stage1_normalized.csv
        raw_data/condition_mapping_data/DXCCSR_v2026-1.csv
Output: 2_condition_mapping/output/stage2_icd10.csv
        2_condition_mapping/output/manual_review_queue.csv

Strategy:
  1. Alias dictionary — exact overrides for frequent/tricky terms (confidence 0.95)
  2. TF-IDF cosine similarity — retrieve top-7 candidates from DXCCSR descriptions
  3. Composite confidence scoring — rank candidates
  4. Auto-accept if best score >= 0.65 and not ambiguous
  5. Flag the rest for manual review

Confidence composite (weights from design review):
  0.40 * token_overlap   (query tokens found in candidate)
  0.30 * jaccard         (symmetric token overlap)
  0.30 * containment     (overlap / smaller token set)
  -0.15 penalty          (candidate name > 2.5x longer than query)

Auto-accept threshold: 0.65
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
STAGE1_OUT   = Path(__file__).parent / "output" / "stage1_normalized.csv"
DXCCSR_PATH  = PROJECT_ROOT / "raw_data" / "condition_mapping_data" / "DXCCSR_v2026-1.csv"
OUTPUT_FILE  = Path(__file__).parent / "output" / "stage2_icd10.csv"
REVIEW_FILE  = Path(__file__).parent / "output" / "manual_review_queue.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUTO_ACCEPT_THRESHOLD = 0.65
TOP_K_CANDIDATES      = 7    # TF-IDF retrieves this many before re-ranking
BATCH_SIZE            = 500  # queries per TF-IDF batch


# ── Alias dictionary ──────────────────────────────────────────────────────────
# Hand-curated overrides for frequent or API-unreliable terms.
# Keys must match condition_normalized values (post Stage 1 normalization).

ALIAS_DICT: dict[str, tuple[str, str]] = {
    # Oncology — hematologic
    "acute myeloid leukemia":                  ("C92.00", "Acute myeloblastic leukemia, not having achieved remission"),
    "acute myeloblastic leukemia":             ("C92.00", "Acute myeloblastic leukemia, not having achieved remission"),
    "acute lymphoblastic leukemia":            ("C91.00", "Acute lymphoblastic leukemia not having achieved remission"),
    "b-cell acute lymphoblastic leukemia":     ("C91.00", "Acute lymphoblastic leukemia not having achieved remission"),
    "chronic lymphocytic leukemia":            ("C91.10", "Chronic lymphocytic leukemia of B-cell type not having achieved remission"),
    "b-cell chronic lymphocytic leukemia":     ("C91.10", "Chronic lymphocytic leukemia of B-cell type not having achieved remission"),
    "chronic myelogenous leukemia":            ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic myeloid leukemia":                ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "diffuse large b-cell lymphoma":           ("C83.30", "Diffuse large B-cell lymphoma, unspecified site, not having achieved remission"),
    "multiple myeloma":                        ("C90.00", "Multiple myeloma not having achieved remission"),
    "myelodysplastic syndromes":               ("D46.9",  "Myelodysplastic syndrome, unspecified"),
    "myelodysplastic syndrome":                ("D46.9",  "Myelodysplastic syndrome, unspecified"),
    "primary myelofibrosis":                   ("D47.1",  "Chronic myeloproliferative disease"),
    "follicular lymphoma":                     ("C82.90", "Follicular lymphoma, unspecified, unspecified site, not having achieved remission"),
    "mantle cell lymphoma":                    ("C83.10", "Mantle cell lymphoma, unspecified site, not having achieved remission"),
    "hodgkin lymphoma":                        ("C81.90", "Hodgkin lymphoma, unspecified, unspecified site, not having achieved remission"),
    "hodgkin disease":                         ("C81.90", "Hodgkin lymphoma, unspecified, unspecified site, not having achieved remission"),
    "non-hodgkin lymphoma":                    ("C85.90", "Non-Hodgkin lymphoma, unspecified, unspecified site, not having achieved remission"),
    "b-cell lymphoma":                         ("C85.10", "Unspecified B-cell lymphoma, unspecified site, not having achieved remission"),
    "t-cell lymphoma":                         ("C84.90", "Mature T/NK-cell lymphomas, unspecified, unspecified site, not having achieved remission"),
    "waldenstrom macroglobulinemia":           ("C88.00", "Waldenstrom macroglobulinemia not having achieved remission"),
    # Oncology — solid tumors
    "breast neoplasms":                        ("C50.919","Malignant neoplasm of unspecified site of unspecified female breast"),
    "breast cancer":                           ("C50.919","Malignant neoplasm of unspecified site of unspecified female breast"),
    "triple negative breast neoplasms":        ("C50.919","Malignant neoplasm of unspecified site of unspecified female breast"),
    "non-small-cell lung carcinoma":           ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "non-small-cell lung cancer":              ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "small cell lung carcinoma":               ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "lung neoplasms":                          ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "colorectal neoplasms":                    ("C18.9",  "Malignant neoplasm of colon, unspecified"),
    "colorectal cancer":                       ("C18.9",  "Malignant neoplasm of colon, unspecified"),
    "colon cancer":                            ("C18.9",  "Malignant neoplasm of colon, unspecified"),
    "rectal cancer":                           ("C20",    "Malignant neoplasm of rectum"),
    "prostatic neoplasms":                     ("C61",    "Malignant neoplasm of prostate"),
    "prostate cancer":                         ("C61",    "Malignant neoplasm of prostate"),
    "ovarian neoplasms":                       ("C56.9",  "Malignant neoplasm of unspecified ovary"),
    "ovarian cancer":                          ("C56.9",  "Malignant neoplasm of unspecified ovary"),
    "pancreatic neoplasms":                    ("C25.9",  "Malignant neoplasm of pancreas, unspecified"),
    "pancreatic cancer":                       ("C25.9",  "Malignant neoplasm of pancreas, unspecified"),
    "hepatocellular carcinoma":                ("C22.0",  "Liver cell carcinoma"),
    "liver cell carcinoma":                    ("C22.0",  "Liver cell carcinoma"),
    "stomach neoplasms":                       ("C16.9",  "Malignant neoplasm of stomach, unspecified"),
    "gastric cancer":                          ("C16.9",  "Malignant neoplasm of stomach, unspecified"),
    "bladder cancer":                          ("C67.9",  "Malignant neoplasm of bladder, unspecified"),
    "urinary bladder neoplasms":               ("C67.9",  "Malignant neoplasm of bladder, unspecified"),
    "uterine cervical neoplasms":              ("C53.9",  "Malignant neoplasm of cervix uteri, unspecified"),
    "cervical cancer":                         ("C53.9",  "Malignant neoplasm of cervix uteri, unspecified"),
    "endometrial neoplasms":                   ("C54.1",  "Malignant neoplasm of endometrium"),
    "melanoma":                                ("C43.9",  "Malignant melanoma of skin, unspecified"),
    "glioblastoma":                            ("C71.9",  "Malignant neoplasm of brain, unspecified"),
    "brain neoplasms":                         ("C71.9",  "Malignant neoplasm of brain, unspecified"),
    "renal cell carcinoma":                    ("C64.9",  "Malignant neoplasm of kidney, except renal pelvis, unspecified"),
    "kidney cancer":                           ("C64.9",  "Malignant neoplasm of kidney, except renal pelvis, unspecified"),
    "thyroid neoplasms":                       ("C73",    "Malignant neoplasm of thyroid gland"),
    "esophageal cancer":                       ("C15.9",  "Malignant neoplasm of esophagus, unspecified"),
    "esophageal neoplasms":                    ("C15.9",  "Malignant neoplasm of esophagus, unspecified"),
    "head and neck cancer":                    ("C14.8",  "Malignant neoplasm of overlapping sites of lip, oral cavity and pharynx"),
    "squamous cell carcinoma":                 ("C44.99", "Other and unspecified malignant neoplasm of skin, unspecified"),
    "cholangiocarcinoma":                      ("C22.1",  "Intrahepatic bile duct carcinoma"),
    "mesothelioma":                            ("C45.9",  "Mesothelioma, unspecified"),
    "sarcoma":                                 ("C49.9",  "Malignant neoplasm of connective and soft tissue, unspecified"),
    "gist":                                    ("C49.9",  "Malignant neoplasm of connective and soft tissue, unspecified"),
    "urothelial carcinoma":                    ("C67.9",  "Malignant neoplasm of bladder, unspecified"),
    "uveal melanoma":                          ("C69.40", "Malignant neoplasm of unspecified choroid"),
    "choroid neoplasms":                       ("C69.40", "Malignant neoplasm of unspecified choroid"),
    # Cardiovascular
    "hypertension":                            ("I10",    "Essential (primary) hypertension"),
    "essential hypertension":                  ("I10",    "Essential (primary) hypertension"),
    "heart failure":                           ("I50.9",  "Heart failure, unspecified"),
    "atrial fibrillation":                     ("I48.91", "Unspecified atrial fibrillation"),
    "coronary artery disease":                 ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    "acute myocardial infarction":             ("I21.9",  "Acute myocardial infarction, unspecified"),
    "myocardial infarction":                   ("I21.9",  "Acute myocardial infarction, unspecified"),
    "stroke":                                  ("I63.9",  "Cerebral infarction, unspecified"),
    "peripheral artery disease":               ("I73.9",  "Peripheral vascular disease, unspecified"),
    "pulmonary hypertension":                  ("I27.20", "Pulmonary hypertension, unspecified"),
    "pulmonary arterial hypertension":         ("I27.21", "Secondary pulmonary arterial hypertension"),
    "deep vein thrombosis":                    ("I82.409","Acute embolism and thrombosis of unspecified deep veins of unspecified lower extremity"),
    "pulmonary embolism":                      ("I26.99", "Other pulmonary embolism without acute cor pulmonale"),
    # Respiratory
    "copd":                                    ("J44.9",  "Chronic obstructive pulmonary disease, unspecified"),
    "asthma":                                  ("J45.909","Unspecified asthma, uncomplicated"),
    "idiopathic pulmonary fibrosis":           ("J84.112","Idiopathic pulmonary fibrosis"),
    "cystic fibrosis":                         ("E84.9",  "Cystic fibrosis, unspecified"),
    "sleep apnea":                             ("G47.30", "Sleep apnea, unspecified"),
    "obstructive sleep apnea":                 ("G47.33", "Obstructive sleep apnea (adult) (pediatric)"),
    # Neurological
    "alzheimer disease":                       ("G30.9",  "Alzheimer's disease, unspecified"),
    "alzheimer's disease":                     ("G30.9",  "Alzheimer's disease, unspecified"),
    "parkinson disease":                       ("G20",    "Parkinson's disease"),
    "parkinson's disease":                     ("G20",    "Parkinson's disease"),
    "multiple sclerosis":                      ("G35",    "Multiple sclerosis"),
    "epilepsy":                                ("G40.909","Epilepsy, unspecified, not intractable, without status epilepticus"),
    "migraine":                                ("G43.909","Migraine, unspecified, not intractable, without status migrainosus"),
    "major depressive disorder":               ("F32.9",  "Major depressive disorder, single episode, unspecified"),
    "depression":                              ("F32.9",  "Major depressive disorder, single episode, unspecified"),
    "schizophrenia":                           ("F20.9",  "Schizophrenia, unspecified"),
    "bipolar disorder":                        ("F31.9",  "Bipolar disorder, unspecified"),
    "anxiety disorder":                        ("F41.9",  "Anxiety disorder, unspecified"),
    "attention deficit hyperactivity disorder":("F90.9",  "Attention-deficit hyperactivity disorder, unspecified type"),
    "adhd":                                    ("F90.9",  "Attention-deficit hyperactivity disorder, unspecified type"),
    "autism spectrum disorder":                ("F84.0",  "Autistic disorder"),
    # Metabolic / Endocrine
    "type 2 diabetes mellitus":                ("E11.9",  "Type 2 diabetes mellitus without complications"),
    "type 1 diabetes mellitus":                ("E10.9",  "Type 1 diabetes mellitus without complications"),
    "diabetes mellitus":                       ("E11.9",  "Type 2 diabetes mellitus without complications"),
    "obesity":                                 ("E66.9",  "Obesity, unspecified"),
    "hyperlipidemia":                          ("E78.5",  "Hyperlipidemia, unspecified"),
    "dyslipidemia":                            ("E78.5",  "Hyperlipidemia, unspecified"),
    "hypothyroidism":                          ("E03.9",  "Hypothyroidism, unspecified"),
    "hyperthyroidism":                         ("E05.90", "Thyrotoxicosis, unspecified, without thyrotoxic crisis or storm"),
    "non-alcoholic fatty liver disease":       ("K76.0",  "Fatty (change of) liver, not elsewhere classified"),
    "non-alcoholic steatohepatitis":           ("K75.81", "Nonalcoholic steatohepatitis (NASH)"),
    "glucose intolerance":                     ("R73.09", "Other abnormal glucose"),
    # Infectious
    "hiv":                                     ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "aids":                                    ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "covid-19":                                ("U07.1",  "COVID-19"),
    "tuberculosis":                            ("A15.9",  "Respiratory tuberculosis, unspecified"),
    "tb":                                      ("A15.9",  "Respiratory tuberculosis, unspecified"),
    "hepatitis b":                             ("B16.9",  "Acute hepatitis B without delta-agent and without hepatic coma"),
    "hepatitis c":                             ("B17.10", "Acute hepatitis C without hepatic coma"),
    "sepsis":                                  ("A41.9",  "Sepsis, unspecified organism"),
    # Musculoskeletal
    "rheumatoid arthritis":                    ("M05.79", "Rheumatoid arthritis with rheumatoid factor of multiple sites without organ or systems involvement"),
    "osteoarthritis":                          ("M19.90", "Unspecified osteoarthritis, unspecified site"),
    "ankylosing spondylitis":                  ("M45.9",  "Ankylosing spondylitis of unspecified sites in spine"),
    "gout":                                    ("M10.9",  "Gout, unspecified"),
    "osteoporosis":                            ("M81.0",  "Age-related osteoporosis without current pathological fracture"),
    "systemic lupus erythematosus":            ("M32.9",  "Systemic lupus erythematosus, unspecified"),
    "fibromyalgia":                            ("M79.7",  "Fibromyalgia"),
    "psoriatic arthritis":                     ("L40.50", "Arthropathic psoriasis, unspecified"),
    # Gastrointestinal
    "crohn disease":                           ("K50.90", "Crohn's disease of small intestine without complications"),
    "crohn's disease":                         ("K50.90", "Crohn's disease of small intestine without complications"),
    "ulcerative colitis":                      ("K51.90", "Ulcerative colitis, unspecified, without complications"),
    "irritable bowel syndrome":                ("K58.9",  "Irritable bowel syndrome without diarrhea"),
    "gastroesophageal reflux":                 ("K21.9",  "Gastro-esophageal reflux disease without esophagitis"),
    # Renal
    "renal insufficiency, chronic":            ("N18.9",  "Chronic kidney disease, unspecified"),
    "chronic kidney disease":                  ("N18.9",  "Chronic kidney disease, unspecified"),
    "acute kidney injury":                     ("N17.9",  "Acute kidney failure, unspecified"),
    # Dermatology
    "dermatitis, atopic":                      ("L20.9",  "Atopic dermatitis, unspecified"),
    "psoriasis":                               ("L40.9",  "Psoriasis, unspecified"),
    "acne vulgaris":                           ("L70.0",  "Acne vulgaris"),
    "dry eye syndromes":                       ("H04.129","Dry eye syndrome of unspecified lacrimal gland"),
    # Eye
    "retinitis pigmentosa":                    ("H35.52", "Pigmentary retinal dystrophy"),
    "glaucoma":                                ("H40.9",  "Unspecified glaucoma"),
    "macular degeneration":                    ("H35.30", "Unspecified macular degeneration"),
    "age-related macular degeneration":        ("H35.30", "Unspecified macular degeneration"),
    "diabetic retinopathy":                    ("E11.319","Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema"),
    "cytomegalovirus retinitis":               ("B25.8",  "Other cytomegaloviral diseases"),
    # Congenital / Pediatric
    "congenital heart defects":                ("Q24.9",  "Congenital malformation of heart, unspecified"),
    "congenital heart disease":                ("Q24.9",  "Congenital malformation of heart, unspecified"),
    "congenital adrenal hyperplasia":          ("E25.0",  "Congenital adrenocortical disorders associated with enzyme deficiency"),
    "lead poisoning":                          ("T56.0X1A","Toxic effects of lead and its compounds, accidental (unintentional), initial encounter"),
    "ophthalmia neonatorum":                   ("P39.1",  "Neonatal conjunctivitis and dacryocystitis"),
    # Abbreviations
    "nsclc":   ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "sclc":    ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    "aml":     ("C92.00", "Acute myeloblastic leukemia, not having achieved remission"),
    "cll":     ("C91.10", "Chronic lymphocytic leukemia of B-cell type not having achieved remission"),
    "nhl":     ("C85.90", "Non-Hodgkin lymphoma, unspecified, unspecified site, not having achieved remission"),
    "dlbcl":   ("C83.30", "Diffuse large B-cell lymphoma, unspecified site, not having achieved remission"),
    "hiv":     ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "aids":    ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "tb":      ("A15.9",  "Respiratory tuberculosis, unspecified"),
    "nafld":   ("K76.0",  "Fatty (change of) liver, not elsewhere classified"),
    "nash":    ("K75.81", "Nonalcoholic steatohepatitis (NASH)"),
    "ibs":     ("K58.9",  "Irritable bowel syndrome without diarrhea"),
    "ibd":     ("K51.90", "Ulcerative colitis, unspecified, without complications"),
    "ms":      ("G35",    "Multiple sclerosis"),
    "als":     ("G12.21", "Amyotrophic lateral sclerosis"),
    "ra":      ("M05.79", "Rheumatoid arthritis with rheumatoid factor of multiple sites without organ or systems involvement"),
    "sle":     ("M32.9",  "Systemic lupus erythematosus, unspecified"),
    "cml":     ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "mds":     ("D46.9",  "Myelodysplastic syndrome, unspecified"),
    "ckd":     ("N18.9",  "Chronic kidney disease, unspecified"),
    "aki":     ("N17.9",  "Acute kidney failure, unspecified"),
    "ild":     ("J84.9",  "Interstitial pulmonary disease, unspecified"),
    "pad":     ("I73.9",  "Peripheral vascular disease, unspecified"),
    "dvt":     ("I82.409","Acute embolism and thrombosis of unspecified deep veins of unspecified lower extremity"),
    "pe":      ("I26.99", "Other pulmonary embolism without acute cor pulmonale"),
    "ipf":     ("J84.112","Idiopathic pulmonary fibrosis"),

    # ── Coverage-review additions (ranked by nct_id impact) ──────────────────
    # HIV variants — ICD-10 uses "Human immunodeficiency virus" not "hiv"
    "hiv infections":                               ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv infection":                                ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv-infection":                                ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv-1 infection":                              ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv-1-infection":                              ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv-2 infection":                              ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv i infection":                              ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv -1 infection":                             ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv 1 infection":                              ("B20",    "Human immunodeficiency virus [HIV] disease"),
    "hiv-1 infection in adults (asymptomatic)":     ("Z21",    "Asymptomatic human immunodeficiency virus [HIV] infection status"),
    "hiv-1 infection (asymptomatic)":               ("Z21",    "Asymptomatic human immunodeficiency virus [HIV] infection status"),
    "hiv-infection (asymptomatic)":                 ("Z21",    "Asymptomatic human immunodeficiency virus [HIV] infection status"),
    "acquired immunodeficiency syndrome":           ("B20",    "Human immunodeficiency virus [HIV] disease"),
    # Cognitive / CNS
    "cognitive dysfunction":                        ("R41.9",  "Unspecified symptoms and signs involving cognitive functions and awareness"),
    "migraine disorders":                           ("G43.909","Migraine, unspecified, not intractable, without status migrainosus"),
    "psychotic disorders":                          ("F29",    "Unspecified nonorganic psychosis"),
    "opioid-related disorders":                     ("F11.10", "Opioid abuse, uncomplicated"),
    "sleep initiation and maintenance disorders":   ("G47.00", "Insomnia, unspecified"),
    "sleep wake disorders":                         ("G47.9",  "Sleep disorder, unspecified"),
    "feeding and eating disorders":                 ("F50.9",  "Eating disorder, unspecified"),
    "delirium":                                     ("F05",    "Delirium due to known physiological condition"),
    "deglutition disorders":                        ("R13.10", "Dysphagia, unspecified"),
    "brain concussion":                             ("S06.0X9A","Concussion with loss of consciousness of unspecified duration, initial encounter"),
    "spinal cord neoplasms":                        ("C72.0",  "Malignant neoplasm of spinal cord"),
    "demyelinating diseases":                       ("G37.9",  "Demyelinating disease of central nervous system, unspecified"),
    "progressive supranuclear palsy":               ("G23.1",  "Progressive supranuclear ophthalmoplegia [Steele-Richardson-Olszewski]"),
    "pick disease of the brain":                    ("G31.01", "Pick's disease"),
    "alcoholism":                                   ("F10.20", "Alcohol dependence, uncomplicated"),
    "psychological stress":                         ("Z73.3",  "Stress, not elsewhere classified"),
    # Cardiovascular
    "ischemic stroke":                              ("I63.9",  "Cerebral infarction, unspecified"),
    "acute coronary syndrome":                      ("I24.9",  "Acute ischemic heart disease, unspecified"),
    "heart arrest":                                 ("I46.9",  "Cardiac arrest, cause unspecified"),
    "cardiac arrest":                               ("I46.9",  "Cardiac arrest, cause unspecified"),
    "venous thromboembolism":                       ("I82.91", "Chronic embolism and thrombosis of unspecified vein"),
    "left ventricular dysfunction":                 ("I50.1",  "Left ventricular failure, unspecified"),
    "left ventricular hypertrophy":                 ("I51.7",  "Cardiomegaly"),
    "carotid artery diseases":                      ("I65.29", "Occlusion and stenosis of unspecified carotid artery"),
    "heart valve diseases":                         ("I38",    "Endocarditis, valve unspecified"),
    "ventricular premature complexes":              ("I49.3",  "Ventricular premature depolarization"),
    "vascular malformations":                       ("Q27.9",  "Congenital malformation of peripheral vascular system, unspecified"),
    "carotid artery stenosis":                      ("I65.29", "Occlusion and stenosis of unspecified carotid artery"),
    "atherosclerotic disease":                      ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    "atherosclerotic coronary vascular disease":    ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    "coronary disease":                             ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    # Respiratory
    "respiratory insufficiency":                    ("J96.00", "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia"),
    "community-acquired pneumonia":                 ("J18.9",  "Pneumonia, unspecified organism"),
    "healthcare-associated pneumonia":              ("J18.9",  "Pneumonia, unspecified organism"),
    "human influenza":                              ("J10.1",  "Influenza due to other identified influenza virus with other respiratory manifestations"),
    "influenza":                                    ("J10.1",  "Influenza due to other identified influenza virus with other respiratory manifestations"),
    "influenza a":                                  ("J09.X2", "Influenza due to identified novel influenza A virus with other respiratory manifestations"),
    "influenza b":                                  ("J10.1",  "Influenza due to other identified influenza virus with other respiratory manifestations"),
    # Infectious
    "coronavirus infections":                       ("B34.2",  "Coronavirus infection, unspecified"),
    "papillomavirus infections":                    ("B97.7",  "Papillomavirus as the cause of diseases classified elsewhere"),
    "epstein-barr virus infections":                ("B27.00", "Gammaherpesviral mononucleosis without complication"),
    "epstein-barr virus infection":                 ("B27.00", "Gammaherpesviral mononucleosis without complication"),
    "ebola hemorrhagic fever":                      ("A98.4",  "Ebola virus disease"),
    "post-acute covid-19 syndrome":                 ("U09.9",  "Post-COVID-19 condition, unspecified"),
    "helicobacter pylori":                          ("B96.81", "Helicobacter pylori [H. pylori] as the cause of diseases classified elsewhere"),
    # Metabolic / Endocrine
    "chronic renal insufficiency":                  ("N18.9",  "Chronic kidney disease, unspecified"),
    "glucose metabolism disorders":                 ("E74.81", "Disorders of glucose transport, unspecified"),
    "lipid metabolism disorders":                   ("E78.5",  "Hyperlipidemia, unspecified"),
    "dyslipidemias":                                ("E78.5",  "Hyperlipidemia, unspecified"),
    "hyperlipoproteinemia type ii":                 ("E78.00", "Pure hypercholesterolemia, unspecified"),
    "hypogonadism":                                 ("E29.1",  "Testicular hypofunction"),
    "primary ovarian insufficiency":                ("E28.39", "Other primary ovarian failure"),
    "thyrotoxicosis":                               ("E05.90", "Thyrotoxicosis, unspecified, without thyrotoxic crisis or storm"),
    "frailty":                                      ("R54",    "Age-related physical debility"),
    "premature birth":                              ("O60.10", "Preterm labor without delivery, unspecified trimester"),
    "polycystic ovary syndrome":                    ("E28.2",  "Polycystic ovarian syndrome"),
    # Oncology
    "hematologic neoplasms":                        ("C96.9",  "Malignant neoplasm of lymphoid, hematopoietic and related tissue, unspecified"),
    "colonic neoplasms":                            ("C18.9",  "Malignant neoplasm of colon, unspecified"),
    "rectal neoplasms":                             ("C20",    "Malignant neoplasm of rectum"),
    "liver neoplasms":                              ("C22.9",  "Malignant neoplasm of liver, unspecified"),
    "glioma":                                       ("C71.9",  "Malignant neoplasm of brain, unspecified"),
    "nasopharyngeal carcinoma":                     ("C11.9",  "Malignant neoplasm of nasopharynx, unspecified"),
    "solid tumors":                                 ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "solid tumor":                                  ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "tumors":                                       ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "neoplasms":                                    ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "malignancy":                                   ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "malignant neoplasms":                          ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "neoplasms malignant":                          ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    "salivary gland neoplasms":                     ("C08.9",  "Malignant neoplasm of major salivary gland, unspecified"),
    "biliary tract neoplasms":                      ("C24.9",  "Malignant neoplasm of biliary tract, unspecified"),
    "bile duct neoplasms":                          ("C22.1",  "Intrahepatic bile duct carcinoma"),
    "digestive system neoplasms":                   ("C26.9",  "Malignant neoplasm of ill-defined sites within the digestive organs"),
    "smoldering multiple myeloma":                  ("C90.00", "Multiple myeloma not having achieved remission"),
    "precursor cell lymphoblastic leukemia-lymphoma":("C91.00","Acute lymphoblastic leukemia not having achieved remission"),
    "transitional cell carcinoma":                  ("C67.9",  "Malignant neoplasm of bladder, unspecified"),
    "plasma cell neoplasms":                        ("C90.00", "Multiple myeloma not having achieved remission"),
    "inflammatory breast neoplasms":                ("C50.919","Malignant neoplasm of unspecified site of unspecified female breast"),
    "fallopian tube neoplasms":                     ("C57.00", "Malignant neoplasm of unspecified fallopian tube"),
    "spinal cord neoplasms":                        ("C72.0",  "Malignant neoplasm of spinal cord"),
    "chronic-phase myeloid leukemia":               ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic phase chronic myeloid leukemia":       ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic myeloid leukemia in chronic phase":    ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic phase-chronic myeloid leukemia":       ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic phase myeloid leukemia":               ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "chronic myeloid leukemia (for part b and c)":  ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "philadelphia-positive myeloid leukemia":       ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "philadelphia positive myeloid leukemia":       ("C92.10", "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission"),
    "noninfiltrating intraductal carcinoma":        ("D05.10", "Intraductal carcinoma in situ of unspecified breast"),
    "second primary neoplasms":                     ("C80.1",  "Malignant (primary) neoplasm, unspecified"),
    # Musculoskeletal
    "sickle cell anemia":                           ("D57.1",  "Sickle-cell disease without crisis"),
    "sickle cell disease":                          ("D57.1",  "Sickle-cell disease without crisis"),
    "idiopathic thrombocytopenic purpura":          ("D69.3",  "Immune thrombocytopenic purpura"),
    "von willebrand diseases":                      ("D68.00", "Von Willebrand disease, unspecified"),
    "bone fractures":                               ("M84.40", "Pathological fracture, unspecified site, initial encounter for fracture"),
    "hip fractures":                                ("S72.001A","Fracture of unspecified part of neck of right femur, initial encounter"),
    "rotator cuff injuries":                        ("M75.100","Unspecified rotator cuff tear or rupture of unspecified shoulder"),
    "patellofemoral pain syndrome":                 ("M25.361","Stiffness of right knee, not elsewhere classified"),
    "inflammatory bowel diseases":                  ("K51.90", "Ulcerative colitis, unspecified, without complications"),
    # GI
    "acute liver failure":                          ("K72.01", "Acute and subacute hepatic failure with coma"),
    "cholestasis":                                  ("K71.0",  "Toxic liver disease with cholestasis"),
    # Miscellaneous high-impact
    "papillomavirus infections":                    ("B97.7",  "Papillomavirus as the cause of diseases classified elsewhere"),
    "eye diseases":                                 ("H57.9",  "Unspecified disorder of eye and adnexa"),
    "macular edema":                                ("H35.81", "Retinal edema"),
    "diabetic macular edema":                       ("E11.3519","Type 2 diabetes mellitus with proliferative diabetic retinopathy with macular edema, unspecified eye"),
    "fuchs' endothelial dystrophy":                 ("H18.51", "Endothelial corneal dystrophy"),
    "turner syndrome":                              ("Q96.9",  "Turner's syndrome, unspecified"),
    "turner's syndrome":                            ("Q96.9",  "Turner's syndrome, unspecified"),
    "congenital diaphragmatic hernias":             ("Q79.0",  "Congenital diaphragmatic hernia"),
    "adenomatous polyposis coli":                   ("K63.5",  "Polyp of colon"),
    "deafness":                                     ("H91.90", "Unspecified hearing loss, unspecified ear"),
    "leg ulcer":                                    ("L97.909","Non-pressure chronic ulcer of unspecified part of unspecified lower leg with unspecified severity"),
    "burns":                                        ("T30.0",  "Burn of unspecified body region, unspecified degree"),
    "bites and stings":                             ("T63.91XA","Toxic effect of contact with unspecified venomous animal, accidental (unintentional), initial encounter"),
    "gingivitis":                                   ("K05.10", "Chronic gingivitis, plaque induced"),
    "deglutition disorders":                        ("R13.10", "Dysphagia, unspecified"),
    "uterine cervical dysplasia":                   ("N87.1",  "Moderate cervical dysplasia"),
    "prostatic hyperplasia":                        ("N40.0",  "Benign prostatic hyperplasia without lower urinary tract symptoms"),
    "bph (benign prostatic hyperplasia)":           ("N40.0",  "Benign prostatic hyperplasia without lower urinary tract symptoms"),
    "polycystic ovary syndrome":                    ("E28.2",  "Polycystic ovarian syndrome"),
    "female genital diseases":                      ("N94.9",  "Unspecified condition associated with female genital organs and menstrual cycle"),
    "pelvic organ prolapse":                        ("N81.9",  "Female genital prolapse, unspecified"),
    "generalized epilepsy":                         ("G40.409","Other generalized epilepsy and epileptic syndromes, not intractable, without status epilepticus"),
    "epileptic syndromes":                          ("G40.409","Other generalized epilepsy and epileptic syndromes, not intractable, without status epilepticus"),
    "neurogenic urinary bladder":                   ("N31.9",  "Neuromuscular dysfunction of bladder, unspecified"),
    "cytopenia":                                    ("D46.A",  "Refractory cytopenia with multilineage dysplasia"),
    "hemolysis":                                    ("D59.9",  "Autoimmune hemolytic anemia, unspecified"),
    "afibrinogenemia":                              ("D68.2",  "Hereditary deficiency of other clotting factors"),
    "hypophosphatemic rickets":                     ("E83.39", "Other disorders of phosphorus metabolism"),
    "legg-calve-perthes disease":                   ("M91.10", "Juvenile osteochondrosis of head of femur [Legg-Calve-Perthes], unspecified leg"),
    "anovulation":                                  ("N97.0",  "Female infertility associated with anovulation"),
    "fecal incontinence":                           ("R15.9",  "Full incontinence of feces"),
    "gestational weight gain":                      ("O26.00", "Excessive weight gain in pregnancy, unspecified trimester"),
    "community-acquired pneumonia":                 ("J18.9",  "Pneumonia, unspecified organism"),
    "exanthema":                                    ("B09",    "Unspecified viral infection characterized by skin and mucous membrane lesions"),
    "croup":                                        ("J05.0",  "Acute obstructive laryngitis [croup]"),
    "ebola hemorrhagic fever":                      ("A98.4",  "Ebola virus disease"),
    "placental insufficiency":                      ("O36.5190","Maternal care for known or suspected placental insufficiency, unspecified trimester, unspecified"),
    "thyrotoxicosis":                               ("E05.90", "Thyrotoxicosis, unspecified, without thyrotoxic crisis or storm"),
    "dystocia":                                     ("O66.0",  "Obstructed labor due to shoulder dystocia"),
    "smoldering multiple myeloma":                  ("C90.00", "Multiple myeloma not having achieved remission"),
    "pseudarthrosis":                               ("M96.0",  "Pseudarthrosis after fusion or arthrodesis"),
    "manic disorder":                               ("F30.9",  "Manic episode, unspecified"),
    "generalized epilepsy":                         ("G40.409","Other generalized epilepsy and epileptic syndromes, not intractable, without status epilepticus"),
    "ventricular premature complexes":              ("I49.3",  "Ventricular premature depolarization"),
    "fuchs' endothelial dystrophy":                 ("H18.51", "Endothelial corneal dystrophy"),
}


# ── Ambiguity detection ───────────────────────────────────────────────────────

AMBIGUOUS_EXACT: frozenset[str] = frozenset({
    # Generic disease words
    "all", "cancer", "tumor", "neoplasm", "disorder", "disease", "syndrome",
    # Symptom-only strings — too vague to map to single ICD-10 code
    "pain", "headache", "fatigue", "fever", "nausea", "stress", "stress test",
    # Healthy volunteer labels — not a disease condition
    "healthy", "healthy volunteers", "healthy volunteer", "normal", "normal volunteers",
    "normal healthy volunteers", "healthy adults", "healthy subjects",
    # Non-medical strings that slipped through
    "safety", "treatment", "tolerance", "patients", "other", "non",
    "balance", "grooming", "altitude", "diagnostic", "antithrombotic",
    "cell therapy", "vaccination", "elderly", "crowding", "pediatric",
    "parent", "literacy", "habits", "health", "smoking",
    # Ambiguous short clinical descriptors
    "aca", "acc", "acm", "abpa", "aatd", "b-cell",
    # Deliberately ambiguous oncology
    "leukemias", "tumors", "malignancy",
})

_AMBIGUOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\w{2,3}$"),
    re.compile(r"^(solid|mixed|multiple)\s+(tumor|cancer|neoplasm)s?$", re.I),
    re.compile(r"^(effect|effects)\s+of\s+\w+", re.I),   # "effect of food", "effects of drugs"
    re.compile(r"^(or|and)\s+\w+", re.I),                # "or cancer", "and disease"
]


def is_ambiguous(query: str) -> bool:
    q = query.strip().lower()
    if q in AMBIGUOUS_EXACT:
        return True
    return any(p.fullmatch(q) for p in _AMBIGUOUS_PATTERNS)


# ── Confidence scoring ────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "and", "or", "of", "the", "with", "in", "a", "an", "not",
    "by", "to", "due", "for", "without", "unspecified",
})


def _tokens(text: str) -> frozenset[str]:
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def composite_confidence(query: str, candidate_name: str) -> float:
    q_norm = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    c_norm = re.sub(r"[^\w\s]", " ", candidate_name.lower()).strip()

    if q_norm == c_norm:
        return 1.0

    q_tok = _tokens(query)
    c_tok = _tokens(candidate_name)

    if not q_tok or not c_tok:
        return 0.0

    inter = q_tok & c_tok
    overlap     = len(inter) / len(q_tok)
    jaccard     = len(inter) / len(q_tok | c_tok)
    containment = len(inter) / min(len(q_tok), len(c_tok))

    score = 0.40 * overlap + 0.30 * jaccard + 0.30 * containment

    len_ratio = len(c_tok) / max(len(q_tok), 1)
    if len_ratio > 2.5:
        score -= 0.15

    return round(min(max(score, 0.0), 1.0), 4)


# ── DXCCSR loader ─────────────────────────────────────────────────────────────

def load_dxccsr() -> pd.DataFrame:
    """Load DXCCSR, strip embedded quotes from column names and values."""
    df = pd.read_csv(DXCCSR_PATH, dtype=str, low_memory=False)
    df.columns = [c.strip("'\" ") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip("'\" ").str.strip()
    df = df[df["ICD-10-CM CODE"].notna() & df["ICD-10-CM CODE DESCRIPTION"].notna()]
    logger.info("DXCCSR: %s ICD-10 codes loaded", f"{len(df):,}")
    return df


# ── TF-IDF index ──────────────────────────────────────────────────────────────

def build_tfidf_index(descriptions: pd.Series) -> tuple:
    """Return (vectorizer, tfidf_matrix) for all ICD-10 descriptions."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(descriptions.str.lower().fillna(""))
    logger.info("TF-IDF index built: %s terms × %s vocab", *map(str, matrix.shape))
    return vectorizer, matrix


def tfidf_top_k(
    vectorizer,
    tfidf_matrix,
    queries: list[str],
    k: int = TOP_K_CANDIDATES,
) -> list[list[int]]:
    """Return top-k DXCCSR row indices for each query (batched for memory)."""
    results: list[list[int]] = []
    q_matrix = vectorizer.transform([q.lower() for q in queries])
    # Compute cosine similarity batch (queries × descriptions)
    sims = cosine_similarity(q_matrix, tfidf_matrix)
    for row in sims:
        top = np.argpartition(row, -k)[-k:]
        top = top[np.argsort(row[top])[::-1]]
        results.append(top.tolist())
    return results


# ── Per-string processing ─────────────────────────────────────────────────────

def process_string(
    query: str,
    top_indices: list[int],
    dxccsr: pd.DataFrame,
) -> dict:
    base: dict = {
        "condition_normalized": query,
        "selected_code":   None,
        "selected_name":   None,
        "confidence":      None,
        "match_tier":      None,
        "status":          None,
        "candidates_json": "[]",
    }
    q = query.strip().lower()

    # Tier 1: alias dictionary
    if q in ALIAS_DICT:
        code, name = ALIAS_DICT[q]
        cand = [{"code": code, "name": name, "confidence": 0.95}]
        base.update(selected_code=code, selected_name=name,
                    confidence=0.95, match_tier="alias",
                    status="auto_accepted",
                    candidates_json=json.dumps(cand))
        return base

    ambiguous = is_ambiguous(q)

    if not top_indices:
        base.update(match_tier="tfidf_no_results", status="no_results")
        return base

    # Score candidates
    scored = []
    for idx in top_indices:
        row = dxccsr.iloc[idx]
        name = row["ICD-10-CM CODE DESCRIPTION"]
        code = row["ICD-10-CM CODE"]
        conf = composite_confidence(query, name)
        scored.append({"code": code, "name": name, "confidence": conf})
    scored.sort(key=lambda x: x["confidence"], reverse=True)

    best = scored[0]
    base["candidates_json"] = json.dumps(scored[:5])
    base["selected_code"]   = best["code"]
    base["selected_name"]   = best["name"]
    base["confidence"]      = best["confidence"]

    if ambiguous or best["confidence"] < AUTO_ACCEPT_THRESHOLD:
        base.update(match_tier="tfidf_flagged", status="manual_review")
    else:
        base.update(match_tier="tfidf_auto", status="auto_accepted")

    return base


# ── Runner ────────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info("Loading Stage 1 output")
    stage1 = pd.read_csv(STAGE1_OUT, low_memory=False)
    unique_strings: list[str] = (
        stage1["condition_normalized"].dropna().unique().tolist()
    )
    logger.info("Unique condition_normalized strings: %s", f"{len(unique_strings):,}")

    # Load DXCCSR and build TF-IDF index
    dxccsr = load_dxccsr()
    vectorizer, tfidf_matrix = build_tfidf_index(dxccsr["ICD-10-CM CODE DESCRIPTION"])

    # Separate alias-handled strings from those needing TF-IDF
    alias_keys = set(ALIAS_DICT.keys())
    need_tfidf = [s for s in unique_strings if s.strip().lower() not in alias_keys]
    alias_only = [s for s in unique_strings if s.strip().lower() in alias_keys]
    logger.info("Alias matches: %s  |  TF-IDF needed: %s",
                f"{len(alias_only):,}", f"{len(need_tfidf):,}")

    # Batch TF-IDF retrieval for all non-alias strings
    logger.info("Running TF-IDF retrieval in batches of %s ...", BATCH_SIZE)
    all_top_indices: list[list[int]] = []
    for i in range(0, len(need_tfidf), BATCH_SIZE):
        batch = need_tfidf[i : i + BATCH_SIZE]
        all_top_indices.extend(tfidf_top_k(vectorizer, tfidf_matrix, batch))
        if (i // BATCH_SIZE + 1) % 5 == 0 or i + BATCH_SIZE >= len(need_tfidf):
            logger.info("  TF-IDF: %s/%s batches done",
                        f"{min(i + BATCH_SIZE, len(need_tfidf)):,}",
                        f"{len(need_tfidf):,}")

    # Process all strings
    logger.info("Scoring candidates ...")
    results: list[dict] = []

    # Alias strings
    for s in alias_only:
        results.append(process_string(s, [], dxccsr))

    # TF-IDF strings
    for s, top_idx in zip(need_tfidf, all_top_indices):
        results.append(process_string(s, top_idx, dxccsr))

    cache_df = pd.DataFrame(results)

    # ── Status summary ───────────────────────────────────────────────────────
    logger.info("Status distribution (unique strings):\n%s",
                cache_df["status"].value_counts().to_string())
    logger.info("Match tier distribution:\n%s",
                cache_df["match_tier"].value_counts().to_string())

    n_auto  = (cache_df["status"] == "auto_accepted").sum()
    n_review = (cache_df["status"] == "manual_review").sum()
    logger.info("Auto-accepted: %.1f%%  |  Manual review needed: %s strings",
                100 * n_auto / len(cache_df), f"{n_review:,}")

    # ── Join with Stage 1 ────────────────────────────────────────────────────
    logger.info("Joining ICD-10 results onto Stage 1 ...")
    join_cols = ["condition_normalized", "selected_code", "selected_name",
                 "confidence", "match_tier", "status"]
    stage2 = stage1.merge(
        cache_df[join_cols].rename(columns={
            "selected_code": "icd10_code",
            "selected_name": "icd10_name",
            "confidence":    "icd10_confidence",
            "match_tier":    "icd10_match_tier",
            "status":        "icd10_status",
        }),
        on="condition_normalized",
        how="left",
    )
    stage2.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved Stage 2 → %s (%s rows)", OUTPUT_FILE, f"{len(stage2):,}")

    # ── Manual review queue ──────────────────────────────────────────────────
    review = (
        cache_df[cache_df["status"] == "manual_review"]
        .sort_values("confidence", ascending=False)
        [["condition_normalized", "selected_code", "selected_name",
          "confidence", "match_tier", "candidates_json"]]
    )
    review.to_csv(REVIEW_FILE, index=False)
    logger.info("Manual review queue → %s (%s strings)", REVIEW_FILE, f"{len(review):,}")

    # Sample of low-confidence strings
    low_conf = cache_df[cache_df["status"] == "manual_review"].head(10)
    if len(low_conf):
        logger.info("Sample manual review strings:")
        for _, r in low_conf.iterrows():
            logger.info("  %r → %s (%.2f)", r["condition_normalized"],
                        r["selected_code"], r["confidence"] or 0)


if __name__ == "__main__":
    run()
