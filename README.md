# credible-clients

This repository contains a small pre-task for potential ML team
members for [UBC Launch Pad](https://www.ubclaunchpad.com).


## Overview

The dataset bundled in this repository contains information about credit card
bill payments, courtesy of the [UCI Machine Learning Repository][UCI]. Your task
is to train a model on this data to predict whether a customer will/won't default
on their next payment. Don't worry about getting "good" results — treat this as an
exploratory task! Feel free to import whatever other libraries you wish.

You should do most of your work in [`main.py`](main.py). It contains some
scaffolding that loads the training data from disk and prints out the accuracy
of the given model. To run this code, you'll need Python and two libraries:
[NumPy] and [`scikit-learn`]. After invoking `python main.py` from your shell
of choice, you should see the model accuracy printed: approximately 50%,
since the provided model predicts completely at random.

## Instructions

Here are the three things you should do:

1. Fork this repo, so we can see your code!
2. Replace the [`DummyClassifier`](main.py#L16-L17) with your own model.
3. Fill in the "write-up" section below in your copy of the README.

_Good luck, and have fun with this_! :rocket:


## Write-up

Give a brief summary of the approach you decided to take, and why! Include
your model accuracy as well!


## Data Format

`X_train` and `X_test` contain data of the following form:

| Column(s) | Data |
| :-------: | ---- |
| 0     | Amount of credit given, in dollars |
| 1     | Gender (_1 = male, 2 = female_) |
| 2     | Education (_1 = graduate school; 2 = university; 3 = high school; 4 = others_) |
| 3     | Marital status (_1 = married; 2 = single; 3 = others_) |
| 4     | Age, in years |
| 5–10  | History of past payments over 6 months (_-1 = on-time; 1 = one month late; …_) |
| 11–16 | Amount of previous bill over 6 months, in dollars |
| 17–22 | Amount of previous payment over 6 months, in dollars |

`y_train` and `y_test` contain a `1` if the customer defaulted on their next
payment, and a `0` otherwise.


[UCI]: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
[NumPy]: http://www.numpy.org
[`scikit-learn`]: http://scikit-learn.org
