from sklearn.ensemble import RandomForestClassifier
from kedro.pipeline import Pipeline, node

import logging


def train_model(train_inputs, train_labels, test_inputs, test_labels):
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(train_inputs, train_labels)
    score = forest_clf.score(test_inputs, test_labels)
    logging.info(f"Score: {score}")
    return score


train_model_node = node(
    func=train_model,
    inputs=["train_inputs", "train_labels", "test_inputs", "test_labels"],
    outputs="score",
    name="train_model_node"
)


def create_pipeline():
    return Pipeline([train_model_node])
