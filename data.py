from utils import load_random, load_covid_drkg, get_adjacency_matrix


def load_data(data_type: str,
              num_features: int,
              num_rel_types: int,
              sparse: bool=True,
              graph_type: str='homogeneous',
              num_nodes: int=1000) -> tuple:
    print('[.] Loading data...')
    if data_type == 'random':
        return load_random(num_features,
                           num_rel_types,
                           sparse,
                           graph_type,
                           num_nodes)

    if 'covid' in data_type:
        entity_emb, entity_ids = load_covid_drkg()
        A = get_adjacency_matrix('./data/drkg.tsv', './data/embed/relations.tsv')

        return entity_emb, A
