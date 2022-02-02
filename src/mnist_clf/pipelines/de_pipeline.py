from keras.datasets import mnist
from kedro.pipeline import Pipeline, node


def load_mnist():
    (train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()

    train_shape, test_shape = train_inputs.shape, test_inputs.shape

    train_inputs = (train_inputs
                    .reshape((train_shape[0], train_shape[1]*train_shape[2]))
                    .astype('float32')/255)

    test_inputs = (test_inputs
                    .reshape((test_shape[0], test_shape[1]*test_shape[2]))
                    .astype('float32')/255)

    return train_inputs, train_labels, test_inputs, test_labels


load_mnist_node = node(
    func=load_mnist,
    inputs=None,
    outputs=["train_inputs", "train_labels", "test_inputs", "test_labels"],
    name="load_mnist_node",
)


def create_pipeline(**kwargs):
    return Pipeline([load_mnist_node])
