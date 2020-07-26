import tensorflow as tf
import numpy as np
from spektral.layers import CrystalConv
from tensorflow.python.keras import Input, Model

from model import RGCNLayer, IRGCNModel
from spektral.layers.ops import sp_matrix_to_sp_tensor


tf.keras.backend.set_floatx('float64')
SINGLE, BATCH, MIXED = 1, 2, 3 # Single, batch, mixed
LAYER_K_, MODES_K_, KWARGS_K_ = 'layer', 'modes', 'kwargs'
batch_size = 32
N = 11
F = 7
S = 3
A = np.zeros((N, N))
X = np.random.normal(size=(N, F))
E = np.random.normal(size=(N, N, S))
E_single  = np.random.normal(size=(N * N, S))


TESTS = [
    {
        LAYER_K_: RGCNLayer,
        MODES_K_: [SINGLE],
        KWARGS_K_: {'channels': 7, 'activation': 'relu',
                    'edges': True,  'sparse': [True]}
    }
]


def _test_single_mode(layer, **kwargs):
    print('Single mode')
    sparse = kwargs.pop('sparse', False)
    A_in = Input(shape=(None,), sparse=sparse)
    X_in = Input(shape=(F,))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X, A]

    if kwargs.pop('edges', None):
        E_in = Input(shape=(S,))
        inputs.append(E_in)
        input_data.append(E_single)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    # model = Model(inputs, output)
    model = IRGCNModel(3)
    model(inputs)
    model.summary()

    output = model(input_data)
    assert output.shape == (N, kwargs['channels'])


def _test_batch_mode(layer, **kwargs):
    print('Batch mode')
    A_batch = np.stack([A] * batch_size)
    X_batch = np.stack([X] * batch_size)

    A_in = Input(shape=(N, N))
    X_in = Input(shape=(N, F))
    inputs = [X_in, A_in]
    input_data = [X_batch, A_batch]
    if kwargs.pop('edges', None):
        E_batch = np.stack([E] * batch_size)
        E_in = Input(shape=(N, N, S))
        inputs.append(E_in)
        input_data.append(E_batch)

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)
    model.summary()

    assert output.shape == (batch_size, N, kwargs['channels'])


def _test_mixed_mode(layer, **kwargs):
    print('Mixed mode')
    sparse = kwargs.pop('sparse', False)
    X_batch = np.stack([X] * batch_size)
    A_in = Input(shape=(N,), sparse=sparse)
    X_in = Input(shape=(N,F))
    inputs = [X_in, A_in]
    if sparse:
        input_data = [X_batch, sp_matrix_to_sp_tensor(A)]
    else:
        input_data = [X_batch, A]

    layer_instance = layer(**kwargs)
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, kwargs['channel'])


def _test_get_config(layer, **kwargs):
    if kwargs.get('edges'):
        kwargs.pop('edges')
    layer_instance = layer(**kwargs)
    config = layer_instance.get_config()
    assert layer(**config)


def test_layers():
    for test in TESTS:
        for mode in test[MODES_K_]:
            if mode == SINGLE:
                if 'sparse' in test[KWARGS_K_]:
                    sparse = test[KWARGS_K_].pop('sparse')
                    for s in sparse:
                        _test_single_mode(test[LAYER_K_],
                                          sparse=s, **test[KWARGS_K_])
                else:
                    _test_single_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == BATCH:
                _test_batch_mode(test[LAYER_K_], **test[KWARGS_K_])
            elif mode == MIXED:
                if 'sparse' in test[KWARGS_K_]:
                    sparse = test[KWARGS_K_].pop('sparse')
                    for s in sparse:
                        _test_mixed_mode(test[LAYER_K_], sparse=s,
                                         **test[KWARGS_K_])
                else:
                    _test_mixed_mode(test[LAYER_K_], **test[KWARGS_K_])
        _test_get_config(test[LAYER_K_], **test[KWARGS_K_])

test_layers()