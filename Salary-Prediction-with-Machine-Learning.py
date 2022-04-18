import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def load():
    data = pd.read_csv("datasets/hitters.csv")
    return data


#####################################
# Keşifçi Veri Analizi
#####################################

# Genel Resim

df = load()
check_df(df) # Salary -> 59 NaN

# Nümerik ve kategorik değişkenleri yakalama

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerik ve kategorik değişkenlerin analizi

for col in cat_cols:
    cat_summary(df, col)
for col in num_cols:
    num_summary(df, col, plot=False)

df.groupby(["League", "Division","NewLeague"]).agg({"Salary":["mean", "median"],
                            "Hits":["mean","median"],
                            "Years":["mean", "median"],
                            "CHits":["mean", "median"],
                            "Assists":["mean", "median"],
                            "Errors":["mean", "median"]})

for col in num_cols:
    target_summary_with_num(df, "Salary", col)

# Aykırı Değer Analizi

for col in num_cols:
    print(col, check_outlier(df, col)) # Aykırı değer bulunmamaktadır.

# Eksik Değer Analizi

na_columns = missing_values_table(df, na_name=True)
missing_vs_target(df, "Salary", na_columns)

# Korelasyon Analizi

(df.corr().sort_values("Salary", ascending=False) > 0.9)

#####################################
# Feature Engineering
#####################################

# Eksik ve aykırı değerler için gerekli işlemler

df["Salary"] = df["Salary"].fillna(df["Salary"].median())
missing_values_table(df)
# Aykırı değer bulunmamaktadır.

# Yeni değişkenler oluşturma

df["New_Years"] = pd.qcut(x=df["Years"], q=4,labels = ["Rookie","Youth","Experienced","Veteran"])
df["New_Avg_Hits"] = df["Hits"] / df["AtBat"] # isabet / vuruş
df["New_Avg_HmRun"] = df["HmRun"] / df["AtBat"] # değerli vuruş / vuruş
df["New_CAvg_Hits"] = df["CHits"] / df["CAtBat"] # kariyer isabet / kariyer vuruş
df["New_HmRun_RBI"] = df["HmRun"] * df["RBI"] # değerli vuruş * koşu yaptırdığı oyuncu
df["New_Rums_Hits"] = df["Runs"] * df["Hits"] # isabet * kazandırdığı sayı

#  Encoding işlemleri

# Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

num_cols = [col for col in num_cols if col not in ["Salary","League","Division","NewLeague","New_Years_Youth","New_Years_Experienced","New_Years_Veteran"]]

# Numerik değişkenler için standartlaştırma

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#  Model oluşturma

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)

mean_squared_error(y, y_pred) # MSE = 66649.80732035986
np.sqrt(mean_squared_error(y, y_pred)) # RMSE = 258.16623969907425
mean_absolute_error(y, y_pred) # MAE = 191.24526496816796
reg_model.score(X, y) # R-KARE = 0.6019214056008161

# With Train-Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_train)

mean_squared_error(y_train, y_pred) # Train MSE = 58473.11496763852
np.sqrt(mean_squared_error(y_train, y_pred)) # Train RMSE = 241.81214809773002
reg_model.score(X_train,y_train) # Train R-KARE = 0.6032469739137809

# 5 Katlı CV RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error"))) # 314.5854126166252