import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class FloodModel:
    def __init__(self, data_path, n_components=0.95):
        self.data_path = data_path
        self.n_components = n_components
        self.df = pd.read_csv(data_path)

    def clean_data(self):
        df = self.df
        # 1. 删除重复值
        df = df.drop_duplicates()

        # 2. 缺失值处理
        missing_ratio = df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > 0.2].index  # 删除缺失比例 > 20% 的列
        df = df.drop(columns=columns_to_drop)

        # 数值型列和分类列分开处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # 数值型特征：用中位数填充缺失值
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # 分类特征：用众数填充缺失值
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # 3. 处理不合理值
        for col in numeric_cols:
            if col in ['rainfall', 'elevation', 'river_distance']:  # 示例列
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        self.df = df

    def prepare_data(self):
        df = self.df

        # 特征工程
        def calculate_flood_risk_score(row):
            return (
                row['rainfall'] * 0.3 +
                (100 - row['elevation']) * 0.2 +
                (1000 - row['river_distance']) * 0.15 +
                row['historical_floods'] * 0.35
            )

        df['flood_risk_score'] = df.apply(calculate_flood_risk_score, axis=1)
        df['rainfall_temp_interaction'] = df['rainfall'] * df['temperature']
        df['rainfall_elevation_ratio'] = df['rainfall'] / (df['elevation'] + 1)

        # 特征选择
        features = [
            'rainfall', 'elevation', 'river_distance', 'historical_floods',
            'temperature', 'flood_risk_score', 'rainfall_temp_interaction'
        ]
        X = df[features]
        y = df['flood_hotspot']

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 数据标准化和降维
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=self.n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        return X_train_pca, X_test_pca, y_train, y_test, scaler, features

    def train_random_forest(self, X_train, y_train):
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        return rf_model

    def train_xgboost(self, X_train, y_train):
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        class_report = classification_report(y_test, model.predict(X_test), output_dict=True)
        cm = confusion_matrix(y_test, model.predict(X_test))
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "class_report": class_report,
            "cm": cm,
            "roc_auc": roc_auc,
            "feature_importance": feature_importance,
            "fpr": fpr,
            "tpr": tpr
        }

    def plot_to_base64(self, plot_func):
        img = io.BytesIO()
        plot_func(img)
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    def generate_plots(self, cm, fpr, tpr, roc_auc, feature_importance):
        def plot_cm(img):
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
            plt.title('Random Forest Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(img, format='png')

        def plot_roc(img):
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.savefig(img, format='png')

        def plot_feature_importance(img):
            plt.figure(figsize=(6, 4))
            plt.barh(range(len(feature_importance)), feature_importance, align='center')
            plt.yticks(range(len(feature_importance)), ['rainfall', 'elevation', 'river_distance', 'historical_floods', 'temperature'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance (Random Forest)')
            plt.savefig(img, format='png')

        cm_url = self.plot_to_base64(plot_cm)
        roc_url = self.plot_to_base64(plot_roc)
        feature_importance_url = self.plot_to_base64(plot_feature_importance)

        return cm_url, roc_url, feature_importance_url
