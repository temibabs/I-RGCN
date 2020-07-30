import tensorflow as tf
from spektral.layers import MessagePassing

from tensorflow.python.keras.layers import Dense


class RGCNConv(MessagePassing):
    def __init__(self,
                 channels: int,
                 features: int,
                 num_rel_types: int=1,
                 activation: str=None,
                 use_bias: bool=False,
                 kernel_initializer='normal',
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
        self.features = features
        self.num_rel_types = num_rel_types
        self.activation_function = tf.nn.relu

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.relation_based_layers = [
            Dense(units=self.channels, use_bias=False, activation=None)
            for _ in range(self.num_rel_types)
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        try:
            X, A, E = self.get_inputs(inputs)
        except AssertionError:
            # Assertion in get_inputs() fails because A is a list
            X, A, E = inputs

        messages_per_type = [self.propagate(X, A[self.i], E, **kwargs)
                             for self.i in range(self.num_rel_types)]
        return self.activation_function(tf.concat(messages_per_type, axis=0))

    def message(self, X, **kwargs):
        X_j = self.get_j(X)
        return self.relation_based_layers[self.i](X_j)

    def get_config(self):
        config = {
            'channels': self.channels,
            'trainable': self.trainable,
            'num_rel_types': self.num_rel_types,
            'features': self.features
        }
        base_config = super().get_config()
        base_config.pop('aggregate')

        return {**base_config, **config}


class IRGCNModel(MessagePassing):
    def __init__(self,
                 num_layers: int,
                 features: int,
                 num_rel_types: int,
                 channels: int,
                 embeddings: int,
                 relation_type: str='small',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.features = features
        self.num_rel_types = num_rel_types
        self.channels = channels
        self.embeddings = embeddings
        self.relation_type = relation_type

    def get_config(self):
        config = {
            'embeddings': self.embeddings,
            'num_rel_types': self.num_rel_types,
            'num_layers': self.num_layers,
            'features': self.features,
            'channels': self.channels,
            'relation_type': self.relation_type
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):

        self.rgcn_layers = [
            RGCNConv(self.channels, self.features, activation=None,
                     num_rel_types=self.num_rel_types)
            for _ in range(self.num_layers)
        ]
        self.mlp = [
            Dense(self.features, activation='relu', use_bias=False)
            for _ in range(2)
        ]

        self.built = True

    def call(self, inputs, **kwargs):
        try:
            X, A, E = self.get_inputs(inputs)
        except AssertionError:
            # Assertion in get_inputs() fails because A is a list
            X, A, E = inputs
        output = X

        # RGCN
        for layer in self.rgcn_layers:
            output = layer([output, A, E])

        # MLP
        if self.relation_type == 'small':
            relations = [self.propagate(output, A[relation], E, **kwargs)
                         for  relation in range(self.num_rel_types)]
            return tf.concat(relations, axis=0)

        return output

    def message(self, X, **kwargs):
        X_i = self.get_i(X)
        X_j = self.get_j(X)
        out = self.mlp[0](tf.concat([X_i, X_j], axis=0))
        return self.mlp[1](out)
