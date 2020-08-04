import csv
import os
import tarfile

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import sparse
from spektral.layers.ops import sp_matrix_to_sp_tensor
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from model import IRGCNModel, MLP


def download_and_extract():
    r"""

    :return:
    """
    import requests

    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/drkg.tar.gz"
    path = "./data/"
    filename = "drkg.tar.gz"
    fn = os.path.join(path, filename)
    if os.path.exists("./data/drkg/drkg.tsv"):
        return

    opener, mode = tarfile.open, 'r:gz'
    os.makedirs(path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(path)
    while True:
        try:
            file = opener(filename, mode)
            try: file.extractall()
            finally: file.close()
            break
        except Exception:
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(filename, 'wb') as writer:
                for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')
    os.chdir(cwd)


def load_model(num_layers: int,
               features: int,
               num_rel_types: int,
               channels: int,
               embeddings: int,
               edge_size: str,
               use_pretrained: str,
               edges: bool=False) -> Model:

    print('[.] Building model...')
    if use_pretrained:
        model = MLP(num_features=features)
    else:
        model = IRGCNModel(num_layers=num_layers,
                           num_rel_types=num_rel_types,
                           channels=channels,
                           embeddings=embeddings,
                           relation_type=edge_size)

    A_in = [Input(shape=(None,), sparse=edge_size) for _ in range(num_rel_types)]
    X_in = Input(shape=(features,))
    inputs = [X_in, A_in]

    if edges:
        E_in = Input(shape=(3,))
        inputs.append(E_in)

    output = model(inputs)
    print('[+] Model has been built')
    print(output.shape)

    return Model(inputs, output)


def load_random(num_features: int,
                num_rel_types: int,
                sparse: bool=True,
                graph_type: str='homogeneous',
                num_nodes: int=1000,
                edges: bool=False):
    N = num_nodes
    F = num_features
    S = 3
    A = [
        np.random.randint(0, 2, (N, N)) for _ in range(num_rel_types)
    ] if graph_type == 'heterogeneous' else np.ones((N, N))
    if sparse:
        try:
            A = sp_matrix_to_sp_tensor(A)
        except TypeError:
            A = [sp_matrix_to_sp_tensor(a) for a in A]

    X = np.random.normal(size=(N, F))
    data = [X, A]
    if edges:
        E = np.random.normal(size=(N * N, S))
        data.append(E)

    print('[+] Data has been loaded.')
    return data


def load_covid_drkg():

    download_and_extract()
    entity_idmap_file = './data/embed/entities.tsv'

    # For the purpose of this task, we're not loading the
    # relation embedding
    # relation_idmap_file = './data/embed/drkg-relations.tsv'

    # Load entity embeddings
    entity_emb = np.load('./data/embed/DRKG_TransE_l2_entity.npy')

    return entity_emb, entity_idmap_file


def load_data_from_file(relation_path: str,
                        entities_file: str,
                        relation_map_file: str,
                        edges: str=False):

    graph_df = read_dataset(relation_path)
    graph_df.columns = ['source', 'relationship', 'destination']

    X = get_embedding(entities_file)
    A = get_adjacency_matrix(relation_path,
                             relation_map_file,
                             entities_file)
    E = read_dataset(entities_file).values if edges else None

    return X, A, E


def get_embedding(entities_file: str) -> np.ndarray:
    return read_dataset(entities_file).values


def get_adjacency_matrix(file_path: str,
                         relation_map_file: str,
                         entity_map_file: str):
    entity_df = read_dataset(entity_map_file)
    entity_df.columns = ['entity', 'id']
    relations_df = read_dataset(relation_map_file)
    relations_df.columns = ['rel', 'rel_id']

    relations_map = {}
    with open(relation_map_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t',
                                fieldnames=['treatment', 'id'])
        for row in reader:
            relations_map[row['treatment']] = int(row['id'])

    adjacency: DataFrame = pd.read_csv(file_path, sep='\t')
    adjacency.columns = ['source', 'relationship', 'destination']
    adjacency = adjacency.loc[
        adjacency['relationship'].isin(list(relations_map.keys()))]
    adjacency = pd.merge(adjacency, entity_df,
                         left_on='source', right_on='entity', how='inner')
    adjacency.drop(['source'], axis=1, inplace=True)
    adjacency = adjacency.rename(columns={'id': 'source_id'})

    adjacency = pd.merge(adjacency, entity_df,
                         left_on='destination', right_on='entity', how='inner')
    adjacency.drop(['destination'], axis=1, inplace=True)
    adjacency = adjacency.rename(columns={'id': 'destination_id'})
    adjacency = pd.merge(adjacency, relations_df,
                         left_on='relationship', right_on='rel', how='inner')

    adjacency: list = [
        (group_id , group[['source_id', 'destination_id']].values)
        for group_id, group in adjacency.groupby('rel_id')
    ]

    return [_to_sparse_matrix(mat) for _, mat in sorted(adjacency)]


def _to_sparse_matrix(mat: np.ndarray,) -> np.ndarray:
    ones = np.array([1.] * mat.shape[0])
    return  sp_matrix_to_sp_tensor(
        sparse.csc_matrix((ones, (mat[:, 0], mat[:, 1]))).toarray())


def read_dataset(path: str):
    if path.split('.')[-1] == 'csv': # CSV
        graph_df = pd.read_csv(path, sep=',', header=None)
    elif path.split('.')[-1] == 'tsv': # TSV
        graph_df = pd.read_csv(path, delimiter='\t', header=None)
    else:
        raise TypeError('Invalid type, allowed types are CSV OR TSV')

    return graph_df
