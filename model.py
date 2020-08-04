import tensorflow as tf
from spektral.layers import MessagePassing

from tensorflow.python.keras.layers import Dense


class RGCNConv(MessagePassing):
    def __init__(self,
                 channels: int,
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
        self.num_rel_types = num_rel_types
        self.activation_function = tf.nn.relu

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.relation_based_weights = [
            self.add_weight(shape=(),
                            initializer='normal',
                            trainable=True)
            # Dense(units=self.channels, use_bias=False, activation=None)
            for _ in range(self.num_rel_types)
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        try:
            X, A, E = self.get_inputs(inputs)
        except AssertionError:
            # Assertion in get_inputs() fails because A is a list
            X, A = inputs

        if self.num_rel_types != len(A):
            raise ValueError('Number of relation types should equal lenght of A')
        messages_per_type = [self.propagate(X, A[self.i], **kwargs)
                             for self.i in range(self.num_rel_types)]
        _sum = 0
        for messages in messages_per_type:
            _sum += messages
        return _sum

    def message(self, X, **kwargs):
        X_j = self.get_j(X)
        return self.relation_based_weights[self.i] * X_j

    def get_config(self):
        config = {
            'channels': self.channels,
            'trainable': self.trainable,
            'num_rel_types': self.num_rel_types
        }
        base_config = super().get_config()
        base_config.pop('aggregate')

        return {**base_config, **config}


class IRGCNModel(MessagePassing):
    def __init__(self,
                 num_layers: int,
                 num_rel_types: int,
                 channels: int,
                 embeddings: int,
                 relation_type: str='small',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        self.channels = channels
        self.embeddings = embeddings
        self.relation_type = relation_type

    def get_config(self):
        config = {
            'embeddings': self.embeddings,
            'num_rel_types': self.num_rel_types,
            'num_layers': self.num_layers,
            'channels': self.channels,
            'relation_type': self.relation_type
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):

        self.rgcn_layers = [
            RGCNConv(self.channels, activation=None,
                     num_rel_types=self.num_rel_types)
            for _ in range(self.num_layers)
        ]

        self.mlp = [
            Dense(1, activation='relu', use_bias=False)
            for _ in range(2)
        ]
        self.built = True

    def call(self, inputs, **kwargs):
        try:
            X, A, E = self.get_inputs(inputs)
        except AssertionError:
            # Assertion in get_inputs() fails because A is a list
            X, A = inputs
        except ValueError:
            X, A, E = inputs
        output = X

        # RGCN
        for layer in self.rgcn_layers:
            output = layer([output, A])

        return output


class MLP(MessagePassing):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.dense1 = Dense(self.num_features, use_bias=False, activation='relu')
        self.dense2 = Dense(self.num_features, use_bias=False, activation='relu')

        self.built = True

    def call(self, inputs, **kwargs):
        try:
            X, A, E = self.get_inputs(inputs)
        except AssertionError:
            # Assertion in get_inputs() fails because A is a list
            X, A = inputs

        self.A = A
        messages_per_type = [self.message(X, **kwargs)
                             for self.i in range(len(A))]

        return messages_per_type

    def message(self, X, **kwargs):
        self.index_i = self.A[self.i].indices[:, 0]
        self.index_j = self.A[self.i].indices[:, 1]
        X_i = self.get_i(X)
        X_j = self.get_j(X)

        return self.dense2(self.dense1(tf.concat([X_i, X_j], axis=0)))
