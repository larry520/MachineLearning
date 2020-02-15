# -*-encoding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn import impute


def load_csvdata(path, filename):
    csv_path = os.path.join(path, filename +".csv")
    return pd.read_csv(csv_path)

housing_path = "datasets" + os.sep + "housing"
housing = load_csvdata(housing_path, "housing")
# print(housing["median_house_value"])
# print(housing["median_house_value"].value_counts())

housing["median_house_value_cat"] = np.ceil(housing["median_house_value"]/100000)
housing["median_house_value_cat"].where(housing["median_house_value_cat"]<5, 5, inplace=True)
# print(housing["median_house_value_cat"].value_counts())

spilt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in spilt.split(housing,housing["median_house_value_cat"]):
    stract_train_set = housing.loc[train_index]
    stract_test_set = housing.loc[test_index]

for set in (stract_train_set, stract_test_set):
    # axis 0 行， 1 列
    set.drop(["median_house_value_cat"],axis=1,inplace=True)

housing = stract_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
#              , s=housing["population"]/100, label="population"
#              , c= "median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
#              )
# plt.legend()

# print(housing.describe())

attributes = ["median_house_value","median_income","total_rooms",
                  "housing_median_age", "latitude"]
# scatter_matrix(housing[attributes],figsize=(12,8))

# 构建新特征
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_households"] = housing["population"] / housing["households"]
corr_matrix = housing.corr()

housing.drop(["people_per_households"], axis=1, inplace=True)

housing_label = housing["median_house_value"].copy()
housing = housing.drop(["median_house_value"], axis=1)


housing_cat = housing["ocean_proximity"]
encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot = housing_cat_1hot.toarray()
housing.drop(["ocean_proximity"], axis=1, inplace=True)
for i, colum in enumerate(list(encoder.classes_)):
    housing[colum] = housing_cat_1hot[:,i]

fill = impute.SimpleImputer(strategy="median")
X = fill.fit_transform(housing)
housing_prepared = pd.DataFrame(X, columns=housing.columns)

#--------------- 开始训练模型------------

def verify(housing_label,housing_prediction, method):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(housing_label, housing_prediction) # 方差
    rmse = np.sqrt(mse) # 标准差
    print(method+":\n方差 mse:\t",mse,"\n标准差 rmse:\t",rmse)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,  housing_label)
housing_prediction = lin_reg.predict(housing_prepared)

verify(housing_label,housing_prediction,"line fit")

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_label)
housing_prediction = tree_reg.predict(housing_prepared)

verify(housing_label, housing_prediction, "tree fit")

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_label)
housing_prediction = forest_reg.predict(housing_prepared)

verify(housing_label, housing_prediction, "forest fit")