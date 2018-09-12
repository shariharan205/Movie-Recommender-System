import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection.validation import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold