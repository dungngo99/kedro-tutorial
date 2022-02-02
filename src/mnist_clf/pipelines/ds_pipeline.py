from sklearn.ensemble import RandomForestClassifier
from kedro.pipeline import Pipeline, node

import logging


def train_model(train_inputs, train_labels, test_inputs, test_labels):
    """
    A function to train the RandomForestClassifier model with the training dataset and compute the test accuracy

    Params: A tuple of four variables, train inputs, train labels, test inputs, and test labels
    Return: A score number
    """
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(train_inputs, train_labels)
    score = forest_clf.score(test_inputs, test_labels)
    logging.info(f"Score: {score}")
    return score


# create a train model node
train_model_node = node(
    func=train_model,
    inputs=["train_inputs", "train_labels", "test_inputs", "test_labels"],
    outputs="score",
    name="train_model_node"
)


def create_pipeline():
    """
    A function to create a Kedro pipeline

    Params: None
    Return: A pipeline object with a list of nodes
    """
    return Pipeline([train_model_node])
