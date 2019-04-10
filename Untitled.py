import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
import numpy as np
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
k=[2,3,1,1]
k=sc.fit_transform(k)
