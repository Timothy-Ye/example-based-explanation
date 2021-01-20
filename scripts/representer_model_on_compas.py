import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import feature_extraction
from sklearn import preprocessing

from representer.representer_model import RepresenterModel

df = pd.read_csv("./data/compas-scores-two-years.csv")

# Filters from mbilalzafar/fair-classification.
df = df.dropna(subset=["days_b_screening_arrest"]) # Dropping missing values.
idx = np.logical_and(df["days_b_screening_arrest"]<=30, df["days_b_screening_arrest"]>=-30)
idx = np.logical_and(idx, df["is_recid"] != -1)
idx = np.logical_and(idx, df["c_charge_degree"] != "O") # F: felony, M: misconduct
idx = np.logical_and(idx, df["score_text"] != "NA")
idx = np.logical_and(idx, np.logical_or(df["race"] == "African-American", df["race"] == "Caucasian"))
df = df[idx]

priors_count = np.reshape(preprocessing.scale(df["priors_count"]), (-1, 1))
age_cat = preprocessing.LabelBinarizer().fit(df["age_cat"]).transform(df["age_cat"])
race = preprocessing.LabelBinarizer().fit(df["race"]).transform(df["race"])
sex = preprocessing.LabelBinarizer().fit(df["sex"]).transform(df["sex"])
c_charge_degree = preprocessing.LabelBinarizer().fit(df["c_charge_degree"]).transform(df["c_charge_degree"])

feature_data = np.hstack((
    priors_count,
    age_cat,
    race,
    sex,
    c_charge_degree
))

target_data = np.reshape(np.array(df["two_year_recid"]), (-1, 1))

train_idxs = range(0, 4278)
test_idxs = range(4278, 5278)

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(7,)),
        tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer="l2"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.load_weights("./output/compas_checkpoint")

num_training_points = 4278
num_test_points = 1000

feature_model = lambda x : x
prediction_network = model

representer_values = np.zeros((num_training_points, num_test_points))

representer_model = RepresenterModel(
    feature_model,
    prediction_network,
    feature_data[train_idxs],
    target_data[train_idxs],
    feature_data[test_idxs],
    model.loss
)

for i in range(num_training_points):
    print("Computing representer values of training point", i, "out of", num_training_points)
    for j in range(num_test_points):
        representer_values[i, j] = representer_model.get_representer_value(i, j)

np.savez(
    "./output/representer_model_on_compas.npz",
    representer_values=representer_values,
)