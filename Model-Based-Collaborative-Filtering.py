import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_curve, auc
from surprise import accuracy, Reader, Dataset
from surprise.model_selection import cross_validate, train_test_split, KFold
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
