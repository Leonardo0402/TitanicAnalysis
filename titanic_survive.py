import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap


class TitleExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.title_mapping = {
            "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty",
            "Don": "Royalty", "Sir": "Royalty", "Dr": "Officer", "Rev": "Officer",
            "the Countess": "Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs",
            "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Royalty",
            "Dona": "Royalty"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Title'] = X['Name'].apply(lambda x: re.search(", (.*?)\.", x).group(1))
        X['Title'] = X['Title'].map(self.title_mapping)
        X['Surname'] = X['Name'].apply(lambda x: x.split(',')[0])
        X['NameLength'] = X['Name'].apply(len)
        return X


class TicketExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['TicketPrefix'] = X['Ticket'].apply(lambda x: re.search("([a-zA-Z]+)", x))
        X['TicketPrefix'] = X['TicketPrefix'].apply(lambda x: x.group(0) if x else 'NoPrefix')
        return X


def preprocess_and_feature_engineering(df):
    df['Cabin'] = df['Cabin'].fillna('U0')
    df['Deck'] = df['Cabin'].map(lambda x: re.search("([a-zA-Z]+)", x).group(0))

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    bins = [0, 7, 12, 30, 870]
    labels = [0, 1, 2, 3]
    df['FareBin'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)

    df['LogFare'] = np.log1p(df['Fare'])

    return df


def load_data():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")
    return train_df, test_df


def main():
    print("Loading data...")
    train_df, test_df = load_data()
    print("Data loaded.")

    feature_pipeline = Pipeline(steps=[
        ('title_extractor', TitleExtractor()),
        ('ticket_extractor', TicketExtractor()),
        ('feature_engineering', FunctionTransformer(preprocess_and_feature_engineering))
    ])

    print("Starting feature extraction...")
    train_df = feature_pipeline.fit_transform(train_df)
    test_df = feature_pipeline.transform(test_df)
    print("Feature extraction completed.")

    knn_imputer = KNNImputer(n_neighbors=5)
    print("Starting KNN imputation...")
    train_df[['Age', 'Fare']] = knn_imputer.fit_transform(train_df[['Age', 'Fare']])
    test_df[['Age', 'Fare']] = knn_imputer.transform(test_df[['Age', 'Fare']])
    print("KNN imputation completed.")

    # 检查是否有缺失值
    if train_df.isnull().sum().sum() > 0:
        print("Train dataset contains NaN values after imputation")
    if test_df.isnull().sum().sum() > 0:
        print("Test dataset contains NaN values after imputation")

    numeric_features = ['Age', 'Fare', 'FamilySize', 'NameLength', 'LogFare']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'FareBin', 'TicketPrefix']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    print("Starting preprocessing...")
    X = preprocessor.fit_transform(train_df)
    y = train_df['Survived']
    X_test = preprocessor.transform(test_df)
    print("Preprocessing completed.")

    # 将数据集分割为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting model training...")
    params = {
        'max_depth': [3, 5, 7, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 15],
        'n_estimators': [50, 100, 150, 200, 250, 300]
    }

    forest = RandomForestClassifier()
    forest_cv = RandomizedSearchCV(estimator=forest, param_distributions=params, cv=5, n_iter=50, random_state=42,
                                   n_jobs=-1)
    forest_cv.fit(X_train, y_train)

    print("Model training completed.")

    # 保存模型和预处理器到文件
    joblib.dump(forest_cv, 'titanic_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Model and preprocessor saved to files.")

    # 使用验证集评估模型
    y_val_pred = forest_cv.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    # 使用 SHAP 解释模型
    explainer = shap.TreeExplainer(forest_cv.best_estimator_)
    X_train_dense = X_train.toarray().astype(float)
    shap_values = explainer.shap_values(X_train_dense)

    plt.figure()
    shap.summary_plot(shap_values, X_train_dense, plot_type="bar")
    plt.title('SHAP Summary Plot (Bar)')
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, X_train_dense)
    plt.title('SHAP Summary Plot')
    plt.show()


if __name__ == "__main__":
    main()
