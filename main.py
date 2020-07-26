from absl import app
from absl import flags

from data import load_data
from model import IRGCNModel
from spektral.layers import GlobalSumPool
FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', 10, 'Number of epochs', lower_bound=0)
flags.DEFINE_integer('num_layers', 3, 'Number of layers', lower_bound=1)

def main(argv):
    print('[.] Loading data...')
    inputs = load_data()
    print('[+] Data has been loaded.')
    model = IRGCNModel(FLAGS.num_layers)
    model.summary()
    print(model.trainable_variables)
    print('[+] Model has been built')

if __name__ == '__main__':
    app.run(main)
