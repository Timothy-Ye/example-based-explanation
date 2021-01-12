import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import feature_extraction
from sklearn import preprocessing

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

model.fit(
    feature_data[train_idxs], target_data[train_idxs], epochs=30, shuffle=False
)

model.save_weights("./output/compas_checkpoint")