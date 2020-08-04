from utils import load_random, load_covid_drkg, get_adjacency_matrix, \
    load_data_from_file


def load_data(data_type: str,
              num_features: int,
              num_rel_types: int,
              sparse: bool=True,
              graph_type: str='homogeneous',
              edges: bool=False,
              num_nodes: int=1000,
              data_path: str=None,
              entities_path: str=None,
              relations_path: str=None) -> tuple:
    print('[.] Loading data...')
    if data_type == 'random':
        return load_random(num_features,
                           num_rel_types,
                           sparse,
                           graph_type,
                           num_nodes,
                           edges)

    elif 'covid' in data_type:
        entity_emb, entity_map = load_covid_drkg()
        A = get_adjacency_matrix('./data/drkg.tsv',
                                 'drkg-relations.tsv',
                                 entity_map)

        return entity_emb, A

    elif data_type == 'custom':
        try:
            X, A, E = load_data_from_file(data_path,
                                          entities_path,
                                          relations_path)
            return X, A, E
        except FileNotFoundError:
            print('[-] Incorrect file path...')
            exit(-1)

    else:
        raise ValueError('[-] data_type can only be one of \'random\','
                         '\'covid\', and \'custom\'.')
