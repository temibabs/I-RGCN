import tensorflow as tf
from spektral.layers import MessagePassing

from tensorflow.python.keras import Model, backend as K, Input
from tensorflow.python.keras.layers import Dense


class RGCNLayer(MessagePassing):
    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(aggregate='sum',
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.channels = self.output_dim = channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint
        )
        self.dense_1 = Dense(self.channels, activation='relu',
                             use_bias=False, **layer_kwargs)
        self.dense_2 = Dense(self.channels, activation='relu',
                             use_bias=False, **layer_kwargs)

        self.built = True

    def message(self, X, E=None):
        X_i = self.get_i(X)
        X_j = self.get_j(X)
        mod = 10
        output = self.dense_1(self.dense_2(K.concatenate((X_i, X_j)))) / mod

        return output

    def get_config(self):
        config = {
            'channels': 7,
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        base_config.pop('aggregate')

        return {**base_config, **config}


class IRGCNModel(Model):
    def __init__(self, depth, features, embeddings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.features = features
        self.embeddings = embeddings
        self.build_model()

    def build(self, input_shape):
        assert len(input_shape) >= 2
        X_in = Input(shape=(self.features,))
        A_in = Input(shape=(None,), sparse=True)
        E_in = Input(shape=(self.embeddings,), dtype=tf.int64)
        output = X_in
        for i in range(self.depth):
            output = RGCNLayer(10, activation='relu')([output, A_in, E_in])

        output = Dense(self.features, activation='relu', use_bias=False)(output)
        output = Dense(self.features, activation='relu', use_bias=False)(output)

        self.model = Model(inputs=[X_in, A_in, E_in], outputs=output)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
