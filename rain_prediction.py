import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import glob
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def predict_rainfall(city_name):
    """
    根据城市气候特点预测降雨情况
    
    参数:
    city_name: 城市名称，如'西安'、'武汉'等
    
    返回:
    预测结果和模型准确率
    """
    print(f"\n开始处理{city_name}的降雨预测...")
    
    # 读取城市气象数据文件
    files = glob.glob(f'{city_name}/{city_name}/*.csv')
    if not files:
        print(f"未找到{city_name}的气象数据文件！")
        return None
    
    dfs = []
    for file in files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file, quotechar='"', low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    # 合并所有数据
    data = pd.concat(dfs, ignore_index=True)
    
    # 数据清洗和预处理
    # 转换日期列为日期时间格式
    data['DATE'] = pd.to_datetime(data['DATE'])
    
    # 提取年份、月份和日期作为特征
    data['YEAR'] = data['DATE'].dt.year
    data['MONTH'] = data['DATE'].dt.month
    data['DAY'] = data['DATE'].dt.day
    
    # 清理数据中的非数值字符
    numeric_cols = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'PRCP']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce')
    
    # 创建降雨标签（PRCP > 0 表示有雨）
    data['RAIN'] = (data['PRCP'] > 0).astype(int)
    
    # 计算历史同期降雨概率
    historical_rain_prob = {}
    for month in range(1, 13):
        for day in range(1, 32):
            month_day_data = data[(data['MONTH'] == month) & (data['DAY'] == day)]
            if len(month_day_data) > 0:
                historical_rain_prob[(month, day)] = month_day_data['RAIN'].mean()
    
    # 添加历史同期降雨概率作为特征
    data['HIST_RAIN_PROB'] = data.apply(
        lambda row: historical_rain_prob.get((row['MONTH'], row['DAY']), 0), 
        axis=1
    )
    
    # 添加季节性特征
    data['SIN_MONTH'] = np.sin(2 * np.pi * data['MONTH'] / 12)
    data['COS_MONTH'] = np.cos(2 * np.pi * data['MONTH'] / 12)
    
    # 根据城市气候特点选择合适的训练月份
    if city_name in ['西安', '洛阳']:  # 北方城市
        print(f"{city_name}属于北方城市，使用3-5月数据训练")
        seasonal_data = data[(data['MONTH'] >= 3) & (data['MONTH'] <= 5)]
    elif city_name in ['武汉', '杭州']:  # 中部/长江流域城市
        print(f"{city_name}属于长江流域城市，使用3-6月数据训练")
        seasonal_data = data[(data['MONTH'] >= 3) & (data['MONTH'] <= 6)]
    elif city_name in ['吐鲁番']:  # 西北干旱区
        print(f"{city_name}属于西北干旱区，使用3-7月数据训练")  # 扩大范围捕捉稀有降水
        seasonal_data = data[(data['MONTH'] >= 3) & (data['MONTH'] <= 7)]
    else:  # 南方城市（婺源、毕节等）
        print(f"{city_name}属于南方城市，使用2-5月数据训练")
        seasonal_data = data[(data['MONTH'] >= 2) & (data['MONTH'] <= 5)]
    
    # 选择特征和目标变量
    features = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MONTH', 'DAY', 
                'HIST_RAIN_PROB', 'SIN_MONTH', 'COS_MONTH']
    
    # 移除包含NaN的特征
    valid_features = []
    for feature in features:
        if feature in seasonal_data.columns and not seasonal_data[feature].isna().all():
            valid_features.append(feature)
    
    X = seasonal_data[valid_features].dropna()
    y = seasonal_data['RAIN'].loc[X.index]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    # 1. Logistic回归
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    log_accuracy = accuracy_score(y_test, log_pred)
    log_f1 = f1_score(y_test, log_pred)
    
    # 2. SVM
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)
    
    # 3. 随机森林
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    # 4. 梯度提升树
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred)
    
    # 创建2026年4月4-6日的预测数据
    prediction_dates = pd.DataFrame({
        'DATE': pd.date_range(start='2026-04-04', end='2026-04-06'),
        'MONTH': 4,
        'DAY': [4, 5, 6]
    })
    
    # 添加季节性特征
    prediction_dates['SIN_MONTH'] = np.sin(2 * np.pi * prediction_dates['MONTH'] / 12)
    prediction_dates['COS_MONTH'] = np.cos(2 * np.pi * prediction_dates['MONTH'] / 12)
    
    # 使用历史同期数据的平均值作为预测特征
    april_4_6_historical = data[(data['MONTH'] == 4) & (data['DAY'] >= 4) & (data['DAY'] <= 6)]
    
    # 为预测日期创建特征
    for feature in valid_features:
        if feature in ['MONTH', 'DAY', 'SIN_MONTH', 'COS_MONTH']:
            continue  # 这些已经设置好了
        elif feature == 'HIST_RAIN_PROB':
            # 添加历史同期降雨概率
            for i, row in prediction_dates.iterrows():
                month, day = row['MONTH'], row['DAY']
                prediction_dates.loc[i, 'HIST_RAIN_PROB'] = historical_rain_prob.get((month, day), 0)
        else:
            # 使用历史同期平均值
            if feature in april_4_6_historical.columns:
                avg_value = april_4_6_historical[feature].mean()
                if not pd.isna(avg_value):
                    prediction_dates[feature] = avg_value
                else:
                    # 如果历史同期没有数据，使用所有数据的平均值
                    prediction_dates[feature] = data[feature].mean()
    
    # 确保预测数据包含所有需要的特征
    for feature in valid_features:
        if feature not in prediction_dates.columns:
            print(f"警告: 预测数据中缺少特征 {feature}，使用0填充")
            prediction_dates[feature] = 0
    
    # 标准化预测特征
    prediction_features = prediction_dates[valid_features]
    prediction_features_scaled = scaler.transform(prediction_features)
    
    # 使用四个模型进行预测
    log_predictions = log_model.predict(prediction_features_scaled)
    log_proba = log_model.predict_proba(prediction_features_scaled)[:, 1]
    
    svm_predictions = svm_model.predict(prediction_features_scaled)
    svm_proba = svm_model.predict_proba(prediction_features_scaled)[:, 1]
    
    rf_predictions = rf_model.predict(prediction_features_scaled)
    rf_proba = rf_model.predict_proba(prediction_features_scaled)[:, 1]
    
    gb_predictions = gb_model.predict(prediction_features_scaled)
    gb_proba = gb_model.predict_proba(prediction_features_scaled)[:, 1]
    
    # 输出预测结果
    results = pd.DataFrame({
        'DATE': prediction_dates['DATE'],
        'Logistic回归': log_predictions,
        'SVM': svm_predictions,
        '随机森林': rf_predictions,
        '梯度提升树': gb_predictions,
        'Logistic概率': log_proba,
        'SVM概率': svm_proba,
        '随机森林概率': rf_proba,
        '梯度提升树概率': gb_proba
    })
    
    # 计算模型权重（基于F1分数）
    total_f1 = log_f1 + svm_f1 + rf_f1 + gb_f1
    log_weight = log_f1 / total_f1
    svm_weight = svm_f1 / total_f1
    rf_weight = rf_f1 / total_f1
    gb_weight = gb_f1 / total_f1
    
    # 添加加权投票结果（基于概率）
    results['加权概率'] = (
        results['Logistic概率'] * log_weight + 
        results['SVM概率'] * svm_weight + 
        results['随机森林概率'] * rf_weight +
        results['梯度提升树概率'] * gb_weight
    )
    
    results['加权投票'] = (results['加权概率'] >= 0.5).astype(int)
    
    # 添加预测置信度
    results['预测置信度'] = abs(results['加权概率'] - 0.5) * 2
    
    # 特征重要性分析
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': valid_features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n{city_name}特征重要性排名:")
        print(feature_importance.head())
    
    print(f"\n{city_name}模型准确率和F1分数:")
    print(f"Logistic回归: 准确率={log_accuracy:.4f}, F1={log_f1:.4f}")
    print(f"SVM: 准确率={svm_accuracy:.4f}, F1={svm_f1:.4f}")
    print(f"随机森林: 准确率={rf_accuracy:.4f}, F1={rf_f1:.4f}")
    print(f"梯度提升树: 准确率={gb_accuracy:.4f}, F1={gb_f1:.4f}")
    
    print(f"\n2026年4月4-6日{city_name}降雨预测结果 (1表示有雨，0表示无雨):")
    display_results = results[['DATE', 'Logistic回归', 'SVM', '随机森林', '梯度提升树', '加权投票', '加权概率', '预测置信度']]
    display_results['加权概率'] = display_results['加权概率'].round(2)
    display_results['预测置信度'] = display_results['预测置信度'].round(2)
    print(display_results)
    
    return {
        'city': city_name,
        'accuracies': {
            'logistic': log_accuracy,
            'svm': svm_accuracy,
            'random_forest': rf_accuracy,
            'gradient_boosting': gb_accuracy
        },
        'f1_scores': {
            'logistic': log_f1,
            'svm': svm_f1,
            'random_forest': rf_f1,
            'gradient_boosting': gb_f1
        },
        'predictions': results,
        'feature_importance': feature_importance if hasattr(rf_model, 'feature_importances_') else None
    }

def visualize_results(all_results):
    """
    可视化所有城市的预测结果
    """
    if not all_results:
        return
    
    # 1. 准确率和F1分数比较
    city_names = [result['city'] for result in all_results]
    log_acc = [result['accuracies']['logistic'] for result in all_results]
    svm_acc = [result['accuracies']['svm'] for result in all_results]
    rf_acc = [result['accuracies']['random_forest'] for result in all_results]
    gb_acc = [result['accuracies']['gradient_boosting'] for result in all_results]
    
    log_f1 = [result['f1_scores']['logistic'] for result in all_results]
    svm_f1 = [result['f1_scores']['svm'] for result in all_results]
    rf_f1 = [result['f1_scores']['random_forest'] for result in all_results]
    gb_f1 = [result['f1_scores']['gradient_boosting'] for result in all_results]
    
    # 准确率比较图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(city_names))
    width = 0.2
    
    plt.bar(x - 1.5*width, log_acc, width, label='Logistic回归')
    plt.bar(x - 0.5*width, svm_acc, width, label='SVM')
    plt.bar(x + 0.5*width, rf_acc, width, label='随机森林')
    plt.bar(x + 1.5*width, gb_acc, width, label='梯度提升树')
    
    plt.xlabel('城市')
    plt.ylabel('准确率')
    plt.title('各城市不同模型准确率比较')
    plt.xticks(x, city_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('城市准确率比较.png')
    
    # F1分数比较图
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - 1.5*width, log_f1, width, label='Logistic回归')
    plt.bar(x - 0.5*width, svm_f1, width, label='SVM')
    plt.bar(x + 0.5*width, rf_f1, width, label='随机森林')
    plt.bar(x + 1.5*width, gb_f1, width, label='梯度提升树')
    
    plt.xlabel('城市')
    plt.ylabel('F1分数')
    plt.title('各城市不同模型F1分数比较')
    plt.xticks(x, city_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('城市F1分数比较.png')
    
    # 2. 降雨预测热力图
    rain_pred = np.zeros((len(city_names), 3))
    rain_prob = np.zeros((len(city_names), 3))
    
    for i, result in enumerate(all_results):
        rain_pred[i] = result['predictions']['加权投票'].values
        rain_prob[i] = result['predictions']['加权概率'].values
    
    # 降雨预测热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(rain_pred, annot=True, cmap='Blues', xticklabels=['4月4日', '4月5日', '4月6日'],
                yticklabels=city_names, cbar_kws={'label': '降雨预测 (0=无雨, 1=有雨)'})
    plt.title('2026年4月4-6日各城市降雨预测')
    plt.tight_layout()
    plt.savefig('城市降雨预测热力图.png')
    
    # 降雨概率热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(rain_prob, annot=True, fmt='.2f', cmap='Blues', xticklabels=['4月4日', '4月5日', '4月6日'],
                yticklabels=city_names, cbar_kws={'label': '降雨概率'})
    plt.title('2026年4月4-6日各城市降雨概率')
    plt.tight_layout()
    plt.savefig('城市降雨概率热力图.png')
    
    # 3. 特征重要性可视化
    # 获取所有城市的特征重要性
    all_features = set()
    for result in all_results:
        if result['feature_importance'] is not None:
            for feature in result['feature_importance']['Feature']:
                all_features.add(feature)
    
    if all_features:
        # 创建特征重要性汇总DataFrame
        feature_importance_summary = pd.DataFrame(index=list(all_features), columns=city_names)
        
        # 填充每个城市的特征重要性
        for i, result in enumerate(all_results):
            if result['feature_importance'] is not None:
                city = result['city']
                for _, row in result['feature_importance'].iterrows():
                    feature_importance_summary.loc[row['Feature'], city] = row['Importance']
        
        # 填充NaN值为0
        feature_importance_summary = feature_importance_summary.fillna(0)
        
        # 计算每个特征的平均重要性
        feature_importance_summary['平均重要性'] = feature_importance_summary.mean(axis=1)
        
        # 按平均重要性排序
        feature_importance_summary = feature_importance_summary.sort_values('平均重要性', ascending=False)
        
        # 绘制特征重要性热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(feature_importance_summary[city_names], annot=True, fmt='.2f', cmap='YlGnBu')
        plt.title('各城市特征重要性比较')
        plt.tight_layout()
        plt.savefig('特征重要性热力图.png')
        
        # 绘制平均特征重要性条形图
        plt.figure(figsize=(10, 8))
        feature_importance_summary['平均重要性'].sort_values().plot(kind='barh')
        plt.xlabel('重要性')
        plt.title('特征平均重要性排名')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('特征平均重要性.png')
        
        print("\n所有城市特征重要性汇总:")
        print(feature_importance_summary[['平均重要性'] + city_names])
    
    # 4. 创建预测结果汇总表格
    summary_table = pd.DataFrame(index=city_names, columns=['4月4日', '4月5日', '4月6日'])
    for i, city in enumerate(city_names):
        for j in range(3):
            if rain_pred[i, j] == 1:
                summary_table.iloc[i, j] = f"有雨 ({rain_prob[i, j]:.2f})"
            else:
                summary_table.iloc[i, j] = f"无雨 ({rain_prob[i, j]:.2f})"
    
    print("\n2026年4月4-6日各城市降雨预测汇总:")
    print(summary_table)
    
    # 保存汇总表格为CSV
    summary_table.to_csv('降雨预测汇总.csv', encoding='utf-8-sig')
    
    return summary_table

def main():
    """
    主函数
    """
    # 所有城市列表
    cities = ['西安', '武汉', '婺源', '洛阳', '毕节', '吐鲁番', '杭州']
    
    # 检查城市数据文件夹是否存在
    available_cities = []
    for city in cities:
        if os.path.exists(f'{city}/{city}'):
            available_cities.append(city)
        else:
            print(f"警告: {city}的数据文件夹不存在，将跳过该城市")
    
    # 存储所有城市的结果
    all_results = []
    
    # 对每个城市进行预测
    for city in available_cities:
        result = predict_rainfall(city)
        if result:
            all_results.append(result)
    
    # 可视化所有城市的预测结果
    if all_results:
        summary = visualize_results(all_results)
        
        # 输出最终预测结果
        print("\n最终预测结果:")
        print("根据多模型集成预测，2026年4月4-6日期间:")
        
        # 统计各城市降雨情况
        rain_days = {}
        for city in summary.index:
            rain_count = sum(1 for day in summary.loc[city] if day.startswith('有雨'))
            rain_days[city] = rain_count
        
        # 按降雨天数排序
        sorted_cities = sorted(rain_days.items(), key=lambda x: x[1], reverse=True)
        
        for city, days in sorted_cities:
            if days == 0:
                print(f"- {city}: 预计三天均无降雨")
            elif days == 3:
                print(f"- {city}: 预计三天均有降雨")
            else:
                rain_dates = []
                for i, day in enumerate(['4月4日', '4月5日', '4月6日']):
                    if summary.loc[city, day].startswith('有雨'):
                        rain_dates.append(day)
                rain_dates_str = '、'.join(rain_dates)
                print(f"- {city}: 预计{rain_dates_str}有降雨")
    
    return all_results

if __name__ == "__main__":
    main()

