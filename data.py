import csv
import pandas as pd
import numpy as np
import sys

from utils import download_and_extract

def load_data():
    COV_disease_list = [
        'Disease::SARS-CoV2 E',
        'Disease::SARS-CoV2 M',
        'Disease::SARS-CoV2 N',
        'Disease::SARS-CoV2 Spike',
        'Disease::SARS-CoV2 nsp1',
        'Disease::SARS-CoV2 nsp10',
        'Disease::SARS-CoV2 nsp11',
        'Disease::SARS-CoV2 nsp12',
        'Disease::SARS-CoV2 nsp13',
        'Disease::SARS-CoV2 nsp14',
        'Disease::SARS-CoV2 nsp15',
        'Disease::SARS-CoV2 nsp2',
        'Disease::SARS-CoV2 nsp4',
        'Disease::SARS-CoV2 nsp5',
        'Disease::SARS-CoV2 nsp5_C145A',
        'Disease::SARS-CoV2 nsp6',
        'Disease::SARS-CoV2 nsp7',
        'Disease::SARS-CoV2 nsp8',
        'Disease::SARS-CoV2 nsp9',
        'Disease::SARS-CoV2 orf10',
        'Disease::SARS-CoV2 orf3a',
        'Disease::SARS-CoV2 orf3b',
        'Disease::SARS-CoV2 orf6',
        'Disease::SARS-CoV2 orf7a',
        'Disease::SARS-CoV2 orf8',
        'Disease::SARS-CoV2 orf9b',
        'Disease::SARS-CoV2 orf9c',
        'Disease::MESH:D045169',
        'Disease::MESH:D045473',
        'Disease::MESH:D001351',
        'Disease::MESH:D065207',
        'Disease::MESH:D028941',
        'Disease::MESH:D058957',
        'Disease::MESH:D006517'
    ]
    treatment = ['Hetionet::CtD::Compound:Disease', 'GNBR::T::Compound:Disease']

    sys.path.insert(1, '../utils')
    download_and_extract()

    drug_list = []
    with open("./data/infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t',
                                fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    entity_idmap_file = './data/embed/entities.tsv'
    relation_idmap_file = './data/embed/relations.tsv'

    # Get drugname/disease name to entity ID mappings
    entity_map = {}
    entity_id_map = {}
    relation_map = {}
    with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t',
                                fieldnames=['name', 'id'])
        for row_val in reader:
            entity_map[row_val['name']] = int(row_val['id'])
            entity_id_map[int(row_val['id'])] = row_val['name']

    with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t',
                                fieldnames=['name', 'id'])
        for row_val in reader:
            relation_map[row_val['name']] = int(row_val['id'])

    # handle the ID mapping
    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_map[drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_map[disease])

    treatment_rid = [relation_map[treat] for treat in treatment]

    return drug_ids, treatment_rid, disease_ids
