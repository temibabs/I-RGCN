from absl import app
from absl import flags

from data import load_data
from loss import IRGCNLoss
from solver import Solver

from utils import get_model


FLAGS = flags.FLAGS

# Integer flags
flags.DEFINE_integer('num_epochs', 20, 'How many rounds to train the model')
flags.DEFINE_integer('num_layers', 3, 'Number of RGCN layers to use')
flags.DEFINE_integer('features', 6, 'Number of features')
flags.DEFINE_integer('num_rel_types', 10, 'Number of different relations')
flags.DEFINE_integer('embeddings', 10, 'Number of embeddings to use')
flags.DEFINE_integer('channels', 10, 'Number of channels')
flags.DEFINE_integer('nodes', 5000, 'How many vertices in the graph?')

# Boolean flags
flags.DEFINE_boolean('edges', False, 'Does the model support edge attributes?')
flags.DEFINE_boolean('sparse', True, 'Sparse graph?')
flags.DEFINE_boolean('use_pretrained', True,
                     'Use Knowledge Graph Embeddings from AWS, these embeddings'
                     ' are based on the Drug Repurposing Knowledge Graph (DRKG)')

# String flags
flags.DEFINE_string('data', 'covid-19', 'Where to get training and test data')
flags.DEFINE_string('data_path', None, 'For loading custom dataset')
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_string('edge_size', 'small', 'Is this a sparse graph? (small or large)')
flags.DEFINE_string('graph_type', 'heterogeneous',
                    'Is graph homogeneous or heterogeneous?')

def main(argv):

    model = get_model(FLAGS.num_layers,
                      FLAGS.features,
                      FLAGS.num_rel_types,
                      FLAGS.channels,
                      FLAGS.embeddings,
                      FLAGS.edge_size,
                      FLAGS.use_pretrained)
    model.summary()
    loss_function = IRGCNLoss()
    solver = Solver(model, loss_function)

    input_data = load_data(data_type=FLAGS.data,
                           num_features=FLAGS.features,
                           num_rel_types=FLAGS.num_rel_types,
                           num_nodes=FLAGS.nodes,
                           graph_type=FLAGS.graph_type,
                           sparse=FLAGS.sparse)

    if FLAGS.mode == 'train':
        solver.train(input_data, FLAGS.num_epochs)
    elif FLAGS.mode == 'test':
        raise NotImplemented


if __name__ == '__main__':
    app.run(main)
