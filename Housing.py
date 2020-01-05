# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import tarfile
from six.moves import urllib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn import impute

# np.set_printoptions(threshold=np.inf)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

DisplayFlag = False

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print("data updated success!")

def load_csvdata(path,filename):
    csv_path = os.path.join(path, filename+".csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    """
    利用随机数分割数据集   随机数种子可保持一致，但数据集更新后数据仍会重新洗牌
    :param data:
    :param test_ratio:
    :return:
    """
    # 设置随机数种子以确保每次调用生成的随机数是一致的
    np.random.seed(42)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloc[n] 输出第n行 n是整数
    # iloc[[n1,n2…]] 输出第n1,n2…行
    return data.iloc[train_indices], data.iloc[test_indices]

import hashlib
def test_set_check(identifier, test_ratio, hash):
    # 取该属性哈希值最后一个字节，0~256区间按比例划分
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    # id_ 标识符,如数据集无标识符，需要构建，常用行索引用作ID
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    # loc[n] 、loc[“d”] 输出第n行或第“d”行。可以索引行号或行标签
    return data.loc[~in_test_set], data.loc[in_test_set]

pass # ------step 1 loading-----------
housing = load_csvdata(HOUSING_PATH,"housing")
# region DataFrame 操作
pass # DataFrame 数据  .head() 查看前五行数据
# print(housing.head())
pass # info() 方法可快速获取数据集的简单描述，如总行数、每个属性的类型及非空值的数量
# print(housing.info())
# """
# [5 rows x 10 columns]
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 20640 entries, 0 to 20639
# Data columns (total 10 columns):
# longitude             20640 non-null float64
# latitude              20640 non-null float64
# housing_median_age    20640 non-null float64
# total_rooms           20640 non-null float64
# total_bedrooms        20433 non-null float64
# population            20640 non-null float64
# households            20640 non-null float64
# median_income         20640 non-null float64
# median_house_value    20640 non-null float64
# ocean_proximity       20640 non-null object
# dtypes: float64(9), object(1)
# memory usage: 1.6+ MB
# """
pass # value_counts()方法查看对应属性下有多少种分类存在，第个分类下有多少个区域
# print(housing["ocean_proximity"].value_counts())
# """
# <1H OCEAN     9136
# INLAND        6551
# NEAR OCEAN    2658
# NEAR BAY      2290
# ISLAND           5
# Name: ocean_proximity, dtype: int64
# """
pass # describe()方法显示各数值属性的摘要 如 非空值数量、均值、方差、各区域占比等
# print(housing.describe())
pass
pass # DataFrame 含有直接画图方法 .hist() 条形直方图 bin 将整个值划分的区间个数 figsize 显示图片尺寸
# housing.hist(bins=50,figsize=(5,5))
# plt.show()
# endregion
pass # ------step 2 构建标识符,分割数据集-----------
# region
split_method = 3
# 使用行索引构建标识符
housing_with_id = housing.reset_index() # adds an 'index' column
# train_set, test_set = split_train_test_by_id(housing_with_id,0.2,'index')

# 使用纬度与经度组合构建标识符 数据集分割稳定，一致性好
# 增加新的列 "id"
housing_with_id['id'] = housing["longitude"]*1000 + housing["latitude"]
if split_method == 1:
    train_set, test_set = split_train_test_by_id(housing_with_id,0.2,'id')

if split_method == 3:
    # sklearn 本身亦封装了分割函数
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# ------------分层抽样----------------------
# print(housing["median_income"].value_counts())

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# print(housing["income_cat"].value_counts())

# 对关键影响属性作为标签分层抽样 确保采样均匀
spilt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in spilt.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# a1 = housing["income_cat"].value_counts()/len(housing)
# a2 = strat_train_set["income_cat"].value_counts()/len(strat_train_set)
# a3 = strat_test_sett["income_cat"].value_counts()/len(strat_test_set)

for set in (strat_train_set, strat_test_set):
    # axis 0 行， 1 列
    set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.copy()
# endregion
pass # DataFrame plot绘图函数的使用
# alpha 透明度
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# # 地理区域分布的可视化
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
#              , s=housing["population"]/100, label="population"
#              , c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
#              )
# plt.legend()
# plt.show()
pass # ------step 3 相关性分析-------------
# region
corr_matrix = housing.corr()
if DisplayFlag:
    print("corr_matrix: ",corr_matrix)

if DisplayFlag:
    """关联矩阵"""
    attributes = ["median_house_value","median_income","total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes],figsize=(12,8))

    housing.plot(kind="scatter", x="median_income", y="median_house_value"
                 ,alpha=0.1)
    plt.show()
# endregion
pass # ------step 4 特征工程 构建新特征,观测相关性-------------
# region
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"]  = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
if DisplayFlag:
    # 观测关键参数与各特征的相关性
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
# endregion
pass # ------step 5 数据准备 数据清理-------------
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

if DisplayFlag:
    """缺失特征处理方法"""
    median = 2
    housing.dropna(subset=["total_bedrooms"])  # option1 放弃缺失属性数据对应的整条数据
    housing.drop("total_bedrooms",axis=1)  # option2 放弃这个属性
    housing["total_bedrooms"].fillna(median)  # option3 填充为某个值

_imputer = impute.SimpleImputer(strategy="median")

# 将数据集文本数据剔除用于计算中位数
housing_num = housing.drop("ocean_proximity",axis=1)
_imputer.fit(housing_num) # 计算中位数并存储到实例变量statistics_中
# temp = imputer.statistics_
# temp = housing_num.median().values
X = _imputer.transform(housing_num)  # X 为一个numpy 数组
# print("X:  ",X)
# Numpy 数组 转 DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# region ----------独热编码-----------------------
# 文本标签转换成数字
TransMethod = 2

housing_cat = housing["ocean_proximity"]

# 文本 -> 整数类别 -> 独热向量 步步为营
if TransMethod == 1:
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    housing_cat_encoded = encoder.fit_transform(housing_cat) # array 1*n
    # 已学习映射
    # print(encoder.classes_)

    # fit_transform 需要二维向量  reshape(-1,1)  1*n 转换成 n*1
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))  # 稀疏矩阵 可当二维矩阵使用
    housing_cat_1hot.toarray()  # 稀疏矩阵转array

# 文本 -> 整数类别 -> 独热向量 一步到位
if TransMethod == 2:
    from sklearn.preprocessing import LabelBinarizer
    # sparse_output 默认为False 输出Numpy数组，设置为True输出稀疏矩阵
    encoder = LabelBinarizer(sparse_output=True)
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    # temp = housing_cat_1hot.toarray()

# endregion

# region  -----------自定义转换器----------
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5 ,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    构建转换类，BaseEstimator 基类可提供两个调参方法 get_params() set_params()
    TransformerMixin 基类可直接得到 fit_transform()方法
    """
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:, population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            # np.c_: 左右行连接，行数相等 np.r_: 上下连接，列连接，列数相等
            return np.c_[X,rooms_per_household,population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """SciKit-Learn 无法处理DataFrame,此类对DataFrame 进行预处理。 选取列，返回NumPy数组"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    """LabelBinarizer 只有2个参数，但pipeline 默认有3个参数"""
    def __init__(self,*args,**kwargs):
        self.encoder = LabelBinarizer(*args,**kwargs)
    def fit(self, X, y=None):
        return self.encoder.fit(X)
    def transform(self, X, y=None):
        return self.encoder.fit_transform(X)
# endregion

# region ----------创建 Pipeline------------2020-1-5
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# temp = housing.to_numpy()  # equal to DataFrame.values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer',impute.SimpleImputer(strategy="median"))
    ,('attribs_adder', CombinedAttributesAdder())
    ,('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
# endregion

# region  ----------连接 Pipeline------------2020-1-5
from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)  # 作用同 list(housing_num.columns)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selectior', DataFrameSelector(num_attribs))
    ,('imputer',impute.SimpleImputer(strategy="median"))
    ,('attribs_adder', CombinedAttributesAdder())
    ,('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs))
    ,('label_binarizer', MyLabelBinarizer(sparse_output=True))
])
#
# selector = DataFrameSelector(cat_attribs)
# housing_cat = selector.fit_transform(housing)
# encoder = LabelBinarizer()
# housing_cat_1hot_ = encoder.fit_transform(housing_cat)

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipelien", num_pipeline)
    ,("cat_pipeline", cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)
temp = housing_prepared.toarray()
# endregion

#----------选择和训练模型------------2020-1-5
# region -------------线性模型-------------
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()  # 创建一个回归的实例
lin_reg.fit(housing_prepared, housing_labels)  # 拟合数据集分别是训练集以及训练标签

some_data = housing.iloc[:5]  # 因为是一个DataFrame数据类型，所以利用iloc的方法进行处理
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.fit_transform(some_data)# 调用full_pipeline对数据进行预处理
some_data_prepared = some_data_prepared.toarray()

if some_data_prepared.shape[1]<lin_reg.coef_.size:
    zero_array = np.zeros([some_data_prepared.shape[0],(lin_reg.coef_.size-some_data_prepared.shape[1])])
    some_data_prepared = np.c_[some_data_prepared,zero_array]

print("LinearRegression predictions:\t", lin_reg.predict(some_data_prepared))  # 调用predict()进行预测  独热特征扩展会导致
print("Labels:\t\t", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions) # 方差
lin_rmse = np.sqrt(lin_mse) # 标准差
print("方差 lin_mse:\t",lin_mse,"\n标准差 lin_rmse:\t",lin_rmse)
# endregion

# region -------------决策树模型-------------
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions) # 方差
tree_rmse = np.sqrt(tree_mse) # 标准差
print("方差 tree_mse:\t",tree_mse,"\n标准差 tree_rmse:\t",tree_rmse)
# endregion



pass


