import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

class FloodModel:
    def __init__(self, connection, table_name):
        self.connection = connection
        self.table_name = table_name
        self.df = self.load_data_from_db()  # 加载数据

    def load_data_from_db(self):
        # 使用现有的数据库连接对象来加载数据
        query = f"SELECT * FROM {self.table_name}"
        df = pd.read_sql(query, self.connection)  # 使用 pandas 从数据库中读取数据
        return df

    def clean_data(self):
        df = self.df
        # 1. 删除重复值
        df = df.drop_duplicates()

        # 2. 缺失值处理
        missing_ratio = df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > 0.2].index  # 删除缺失比例 > 20% 的列
        df = df.drop(columns=columns_to_drop)

        # 将所有特征列转换为数值类型（若存在非数值的字符，转换为 NaN）
        numeric_cols = [
            '降水量', '地形坡度', '植被覆盖率', '高楼遮挡', '路面材料透水性',
            '地下水位高度', '附近水体水位', '蒸发能力', '自然排水路径', 
            '排水设施维护状况', '地下管道系统负载', '施工或改建工程',
            '校园垃圾和污染物', '城市排水系统关联性'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 强制转换，无法转换的值变为 NaN
        # 如果列中包含无法转换为数字的文本，将其替换为 NaN
        df['其他因素'] = pd.to_numeric(df['其他因素'], errors='coerce')

        # 处理转换后的缺失值
        df = df.fillna(df.median())  # 对于所有列使用中位数填充缺失值

        self.df = df

    def calculate_flood_risk_score(self, row):
        # 确保每个值都是数值类型
        return (
            row['降水量'] * 0.2 +
            (100 - row['地形坡度']) * 0.1 +
            row['植被覆盖率'] * 0.1 +
            row['高楼遮挡'] * 0.05 +
            row['路面材料透水性'] * 0.1 +
            row['地下水位高度'] * 0.15 +
            row['附近水体水位'] * 0.1 +
            row['蒸发能力'] * 0.05 +
            row['自然排水路径'] * 0.1 +
            row['排水设施维护状况'] * 0.05 +
            row['地下管道系统负载'] * 0.05 +
            row['施工或改建工程'] * 0.05 +
            row['校园垃圾和污染物'] * 0.05
        )

    def prepare_data(self):
        df = self.df
        # 特征工程，计算洪水风险评分
        df['洪水风险评分'] = df.apply(self.calculate_flood_risk_score, axis=1)

        # 特征选择
        features = [
            '降水量', '地形坡度', '植被覆盖率', '高楼遮挡', '路面材料透水性',
            '地下水位高度', '附近水体水位', '蒸发能力', '自然排水路径', 
            '排水设施维护状况', '地下管道系统负载', '施工或改建工程',
            '校园垃圾和污染物', '城市排水系统关联性', '洪水风险评分'
        ]

        # 目标变量：生成一个新的目标变量，根据“洪水风险评分”来进行二分类
        df['洪水热点'] = (df['洪水风险评分'] > 50).astype(int)  # 假设风险评分大于50表示有洪水热点

        X = df[features]
        y = df['洪水热点']

        # 数据划分
        if len(df) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y  # 如果数据只有1行，直接用作训练和测试

        # 数据标准化和降维
        df = df.fillna(df.median())  
        X = df[features]
        X = X.fillna(X.median()) 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=0.95)  # 保持95%的信息
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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['无洪水', '有洪水'], yticklabels=['无洪水', '有洪水'])
            plt.title('随机森林混淆矩阵')
            plt.xlabel('预测')
            plt.ylabel('实际')
            plt.savefig(img, format='png')

        def plot_roc(img):
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('接收者操作特征曲线（ROC）')
            plt.legend(loc='lower right')
            plt.savefig(img, format='png')

        def plot_feature_importance(img):
            plt.figure(figsize=(6, 4))
            plt.barh(range(len(feature_importance)), feature_importance, align='center')
            plt.yticks(range(len(feature_importance)), ['降水量', '地形坡度', '植被覆盖率', '高楼遮挡', '路面材料透水性', '地下水位高度', '附近水体水位', 
                                                       '蒸发能力', '自然排水路径', '排水设施维护状况', '地下管道系统负载', '施工或改建工程', 
                                                       '校园垃圾和污染物', '城市排水系统关联性'])
            plt.xlabel('特征重要性')
            plt.title('随机森林特征重要性')
            plt.savefig(img, format='png')

        cm_url = self.plot_to_base64(plot_cm)
        roc_url = self.plot_to_base64(plot_roc)
        feature_importance_url = self.plot_to_base64(plot_feature_importance)

        return cm_url, roc_url, feature_importance_url
