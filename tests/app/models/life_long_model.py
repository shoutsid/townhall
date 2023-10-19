import numpy as np
from app.models.life_long_model import LifeLongModel


def test_forward_pass():
    """
    Test the forward pass of the LifeLongModel.

    Creates a LifeLongModel instance with 3 input features, passes an input array of shape (3,) to the model's forward
    method, and checks that the output has shape (5,).
    """
    model = LifeLongModel(num_features=3)
    x = np.array([1, 2, 3])
    y_pred = model.forward(x)
    assert y_pred.shape == (5,)


def test_train():
    """
    Test the train method of the LifeLongModel class.

    The function creates an instance of the LifeLongModel class with 3 features, and trains the model
    using a numpy array of shape (3,) and a list of length 5. The function then checks if the loss returned
    by the train method is an instance of numpy.ndarray.
    """
    model = LifeLongModel(num_features=3)
    x = np.array([1, 2, 3])
    y_true = [1, 0, 1, 0, 1]
    loss = model.train(x, y_true)
    assert isinstance(loss, np.ndarray)
