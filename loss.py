import tensorflow as tf
from tensorflow.python.keras.losses import Loss


class IRGCNLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return tf.math.log(1 + tf.math.exp(tf.matmul(-1 * y_true, y_pred)))
