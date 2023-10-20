"""
This module contains the LifeLongModel class, which represents a
lifelong learning model that can be trained on multiple tasks.
"""

from typing import List
from tinygrad.tensor import Tensor
import numpy as np


class LifelongLearningModel:
    """
    A class representing a lifelong learning model.

    Attributes:
    -----------
    num_features : int
        The number of features in the input data.
    w1 : Tensor
        The weight matrix for the first layer of the model.
    w2 : Tensor
        The weight matrix for the second layer of the model.

    Methods:
    --------
    forward(x: np.ndarray) -> np.ndarray
        Performs a forward pass through the model.
    train(x: Tensor, y_true: List[float]) -> float
        Trains the model on a single input-output pair.
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.w1 = Tensor(np.random.randn(5, num_features), requires_grad=True)
        self.w2 = Tensor(np.random.randn(5), requires_grad=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the life_long_model.

        Args:
            x (numpy.ndarray): Input tensor of shape (n_features,).

        Returns:
            numpy.ndarray: Output tensor of shape (n_classes,).
        """
        x = Tensor(x).reshape(1, -1)
        x = self.w1.dot(x.transpose()).relu()
        return self.w2.dot(x)

    def train(self, x: Tensor, y_true: List[float]) -> float:
        """
        Trains the model on the given input and target output.

        Args:
            x (Tensor): Input tensor.
            y_true (list): Target output.

        Returns:
            float: The loss value as a numpy array.
        """
        y_pred = self.forward(x)
        loss = ((y_pred - Tensor(y_true)) ** 2).sum()
        loss.backward()
        learning_rate = 0.01
        self.w1 = self.w1 - self.w1.grad * learning_rate
        self.w2 = self.w2 - self.w2.grad * learning_rate
        return loss.numpy()
