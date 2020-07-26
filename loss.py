import tensorflow as tf
from tensorflow.python.keras.losses import Loss


class IRGCNLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        x = tf.transpose(y_pred[0])* tf.linalg.diag(y_pred[1]) * y_pred[2]
        return tf.math.log(1 + tf.math.exp(tf.matmul(-1 * y_true, x)))
