from keras.datasets import mnist
from kedro.pipeline import Pipeline, node


def load_mnist():
    """
    A function to load and transoform the image data to 2D matrix
    
    Params: None
    Return: A tuple of four variables, train inputs, train labels, test inputs, and test labels
    """
    (train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()

    train_shape, test_shape = train_inputs.shape, test_inputs.shape

    train_inputs = (train_inputs
                    .reshape((train_shape[0], train_shape[1]*train_shape[2]))
                    .astype('float32')/255)

    test_inputs = (test_inputs
                    .reshape((test_shape[0], test_shape[1]*test_shape[2]))
                    .astype('float32')/255)

    return train_inputs, train_labels, test_inputs, test_labels

# create a Kedro node
load_mnist_node = node(
    func=load_mnist,
    inputs=None,
    outputs=["train_inputs", "train_labels", "test_inputs", "test_labels"],
    name="load_mnist_node",
)


def create_pipeline(**kwargs):
    """
    A function to create a Kedro pipeline
    
    Params: None
    Return: A pipeline object with a list of nodes
    """
    return Pipeline([load_mnist_node])
