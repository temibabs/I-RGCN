import time
from numpy import ndarray
from tensorflow.python.keras.models import Model
from typing import List, Tuple, Optional

from tensorflow.python.keras.optimizers import Adam

from loss import IRGCNLoss
from model import MLP

A = List[ndarray]
Input = Tuple[ndarray, A, Optional[ndarray]]


class Solver:
    def __init__(self, model: Model, loss: IRGCNLoss, small: bool=True):
        self.model = model
        self.loss = loss
        self.small = small

        if small:
            self.model2 = MLP(num_features=50)

    def train(self, data: Input, epochs: int):
        optimizer = Adam()
        output = None
        _, A = data
        for epoch in range(epochs):
            node_embedings = self.model(data)
            _input = [node_embedings, A]
            if self.small:
                relation_embeddings = self.model2(_input)
            loss = self.loss(y_true=output, y_pred=output)

    def test(self):
        raise NotImplemented

    def log(self, message: str):
        if self.file_log:
            with open(self.log_directory, 'w+') as f:
                f.write(message)

        print(f'{time.time()}: {message}')
