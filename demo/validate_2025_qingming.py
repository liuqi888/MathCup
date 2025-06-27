import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def validate_2025_qingming(cities):
    """
    验证模型对2025年清明节天气预测的准确性
    
    参数:
    cities: 城市列表
    
    返回:
    验证结果
    """
    print("\n开始验证模型对2025年清明节天气预测的准确性...")
    
    # 2025年清明节日期：4月5日
    # 我们验证4月4日-6日三天的预测
    validation_dates = pd.date_range(start='2025-04-04', end='2025-04-06')
    
    # 存储所有城市的验证结果
    validation_results = []
    city_predictions = {}
    
    for city in cities:
        print(f"\n验证{city}的降雨预测...")
        
        # 读取城市气象数据文件
        files = glob.glob(f'{city}/{city}/*.csv')
        if not files:
            print(f"未找到{city}的气象数据文件！")
            continue
        
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
        
        # 将数据分为训练集（2024年及以前）和验证集（2025年）
        train_data = data[data['YEAR'] < 2025]
        validation_data = data[(data['YEAR'] == 2025) & 
                              (data['MONTH'] == 4) & 
                              (data['DAY'] >= 4) & 
                              (data['DAY'] <= 6)]
        
        if validation_data.empty:
            print(f"警告: 没有找到{city}2025年4月4-6日的实际观测数据，无法验证")
            continue
        
        # 根据城市气候特点选择合适的训练月份
        if city in ['西安', '洛阳']:  # 北方城市
            print(f"{city}属于北方城市，使用3-5月数据训练")
            seasonal_data = train_data[(train_data['MONTH'] >= 3) & (train_data['MONTH'] <= 5)]
        elif city in ['武汉', '杭州']:  # 中部/长江流域城市
            print(f"{city}属于长江流域城市，使用3-6月数据训练")
            seasonal_data = train_data[(train_data['MONTH'] >= 3) & (train_data['MONTH'] <= 6)]
        elif city in ['吐鲁番']:  # 西北干旱区
            print(f"{city}属于西北干旱区，使用3-7月数据训练")
            seasonal_data = train_data[(train_data['MONTH'] >= 3) & (train_data['MONTH'] <= 7)]
        else:  # 南方城市（婺源、毕节等）
            print(f"{city}属于南方城市，使用2-5月数据训练")
            seasonal_data = train_data[(train_data['MONTH'] >= 2) & (train_data['MONTH'] <= 5)]
        
        # 选择特征和目标变量
        features = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MONTH', 'DAY', 
                    'HIST_RAIN_PROB', 'SIN_MONTH', 'COS_MONTH']
        
        # 移除包含NaN的特征
        valid_features = []
        for feature in features:
            if feature in seasonal_data.columns and not seasonal_data[feature].isna().all():
                valid_features.append(feature)
        
        X_train = seasonal_data[valid_features].dropna()
        y_train = seasonal_data['RAIN'].loc[X_train.index]
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 训练模型
        # 1. Logistic回归
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train_scaled, y_train)
        
        # 2. SVM
        svm_model = SVC(random_state=42, probability=True)
        svm_model.fit(X_train_scaled, y_train)
        
        # 3. 随机森林
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # 4. 梯度提升树
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        
        # 准备验证数据
        X_val = validation_data[valid_features].dropna()
        y_val = validation_data['RAIN'].loc[X_val.index]
        
        if X_val.empty:
            print(f"警告: {city}的验证数据中缺少必要特征，无法验证")
            continue
        
        # 标准化验证特征
        X_val_scaled = scaler.transform(X_val)
        
        # 使用模型进行预测
        log_pred = log_model.predict(X_val_scaled)
        log_proba = log_model.predict_proba(X_val_scaled)[:, 1]
        
        svm_pred = svm_model.predict(X_val_scaled)
        svm_proba = svm_model.predict_proba(X_val_scaled)[:, 1]
        
        rf_pred = rf_model.predict(X_val_scaled)
        rf_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
        
        gb_pred = gb_model.predict(X_val_scaled)
        gb_proba = gb_model.predict_proba(X_val_scaled)[:, 1]
        
        # 计算模型权重（基于训练集的F1分数）
        log_cv_pred = cross_val_predict(log_model, X_train_scaled, y_train, cv=5)
        svm_cv_pred = cross_val_predict(svm_model, X_train_scaled, y_train, cv=5)
        rf_cv_pred = cross_val_predict(rf_model, X_train_scaled, y_train, cv=5)
        gb_cv_pred = cross_val_predict(gb_model, X_train_scaled, y_train, cv=5)
        
        log_f1 = f1_score(y_train, log_cv_pred)
        svm_f1 = f1_score(y_train, svm_cv_pred)
        rf_f1 = f1_score(y_train, rf_cv_pred)
        gb_f1 = f1_score(y_train, gb_cv_pred)
        
        # 计算模型权重
        total_f1 = log_f1 + svm_f1 + rf_f1 + gb_f1
        log_weight = log_f1 / total_f1
        svm_weight = svm_f1 / total_f1
        rf_weight = rf_f1 / total_f1
        gb_weight = gb_f1 / total_f1
        
        # 计算加权概率
        weighted_proba = (
            log_proba * log_weight + 
            svm_proba * svm_weight + 
            rf_proba * rf_weight +
            gb_proba * gb_weight
        )
        
        weighted_pred = (weighted_proba >= 0.5).astype(int)
        
        # 计算评估指标
        accuracy = accuracy_score(y_val, weighted_pred)
        precision = precision_score(y_val, weighted_pred, zero_division=0)
        recall = recall_score(y_val, weighted_pred, zero_division=0)
        f1 = f1_score(y_val, weighted_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_val, weighted_pred)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'DATE': validation_data['DATE'].loc[X_val.index],
            '实际降雨': y_val,
            'Logistic回归': log_pred,
            'SVM': svm_pred,
            '随机森林': rf_pred,
            '梯度提升树': gb_pred,
            '加权预测': weighted_pred,
            '加权概率': weighted_proba.round(2)
        })
        
        # 输出验证结果
        print(f"\n{city}2025年清明节降雨预测验证结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("\n预测详情:")
        print(results)
        
        # 存储验证结果
        validation_results.append({
            'city': city,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': results
        })
        
        # 存储城市预测结果
        city_predictions[city] = results
    
    # 可视化验证结果
    if validation_results:
        visualize_validation_results(validation_results, city_predictions)
    
    return validation_results

def visualize_validation_results(validation_results, city_predictions):
    """
    可视化验证结果
    """
    # 提取城市名称和评估指标
    city_names = [result['city'] for result in validation_results]
    accuracies = [result['accuracy'] for result in validation_results]
    precisions = [result['precision'] for result in validation_results]
    recalls = [result['recall'] for result in validation_results]
    f1_scores = [result['f1'] for result in validation_results]
    
    # 创建评估指标条形图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(city_names))
    width = 0.2
    
    plt.bar(x - 1.5*width, accuracies, width, label='准确率')
    plt.bar(x - 0.5*width, precisions, width, label='精确率')
    plt.bar(x + 0.5*width, recalls, width, label='召回率')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1分数')
    
    plt.xlabel('城市')
    plt.ylabel('评分')
    plt.title('2025年清明节降雨预测验证结果')
    plt.xticks(x, city_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('2025年验证结果.png')
    
    # 创建预测vs实际的对比表格
    comparison_table = pd.DataFrame(index=city_names, columns=['4月4日预测', '4月4日实际', 
                                                             '4月5日预测', '4月5日实际', 
                                                             '4月6日预测', '4月6日实际'])
    
    for city in city_names:
        preds = city_predictions[city]
        
        for day in range(4, 7):
            day_preds = preds[preds['DATE'].dt.day == day]
            if not day_preds.empty:
                pred_rain = '有雨' if day_preds['加权预测'].iloc[0] == 1 else '无雨'
                actual_rain = '有雨' if day_preds['实际降雨'].iloc[0] == 1 else '无雨'
                comparison_table.loc[city, f'4月{day}日预测'] = pred_rain
                comparison_table.loc[city, f'4月{day}日实际'] = actual_rain
    
    print("\n2025年清明节降雨预测与实际对比:")
    print(comparison_table)
    
    # 保存对比表格为CSV
    comparison_table.to_csv('2025年清明节降雨预测验证.csv', encoding='utf-8-sig')
    
    # 创建热力图显示预测准确性
    plt.figure(figsize=(15, 8))
    
    # 创建一个新的DataFrame用于热力图，确保数据类型为数值型
    heatmap_data = pd.DataFrame(index=city_names, columns=['4月4日', '4月5日', '4月6日'])
    heatmap_data = heatmap_data.astype('float')  # 确保数据类型为浮点型
    
    for city in city_names:
        preds = city_predictions[city]
        
        for day in range(4, 7):
            day_preds = preds[preds['DATE'].dt.day == day]
            if not day_preds.empty:
                pred = day_preds['加权预测'].iloc[0]
                actual = day_preds['实际降雨'].iloc[0]
                # 1表示预测正确，0表示预测错误
                heatmap_data.loc[city, f'4月{day}日'] = 1.0 if pred == actual else 0.0
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar=False, 
                linewidths=.5, fmt='.0f', vmin=0, vmax=1)
    plt.title('2025年清明节降雨预测准确性 (绿色=正确, 红色=错误)')
    plt.tight_layout()
    plt.savefig('2025年预测准确性热力图.png')
    
    # 计算总体评估指标
    total_pred = []
    total_actual = []
    
    for city in city_names:
        preds = city_predictions[city]
        total_pred.extend(preds['加权预测'].values)
        total_actual.extend(preds['实际降雨'].values)
    
    total_accuracy = accuracy_score(total_actual, total_pred)
    total_precision = precision_score(total_actual, total_pred, zero_division=0)
    total_recall = recall_score(total_actual, total_pred, zero_division=0)
    total_f1 = f1_score(total_actual, total_pred, zero_division=0)
    
    print("\n总体评估指标:")
    print(f"准确率: {total_accuracy:.4f}")
    print(f"精确率: {total_precision:.4f}")
    print(f"召回率: {total_recall:.4f}")
    print(f"F1分数: {total_f1:.4f}")
    
    # 创建总体评估指标图表
    plt.figure(figsize=(8, 6))
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [total_accuracy, total_precision, total_recall, total_f1]
    
    plt.bar(metrics, values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.title('2025年清明节降雨预测总体评估指标')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上添加具体数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('2025年总体评估指标.png')
    
    # 创建模型比较图表
    plt.figure(figsize=(14, 8))
    
    # 计算每个模型的准确率
    model_accuracies = {
        'Logistic回归': [],
        'SVM': [],
        '随机森林': [],
        '梯度提升树': [],
        '加权预测': []
    }
    
    for city in city_names:
        preds = city_predictions[city]
        actual = preds['实际降雨'].values
        
        for model in ['Logistic回归', 'SVM', '随机森林', '梯度提升树', '加权预测']:
            model_pred = preds[model].values
            model_acc = accuracy_score(actual, model_pred)
            model_accuracies[model].append(model_acc)
    
    # 绘制每个模型的准确率
    x = np.arange(len(city_names))
    width = 0.15
    
    for i, (model, accs) in enumerate(model_accuracies.items()):
        plt.bar(x + (i - 2) * width, accs, width, label=model)
    
    plt.xlabel('城市')
    plt.ylabel('准确率')
    plt.title('各模型在不同城市的预测准确率比较')
    plt.xticks(x, city_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('2025年模型比较.png')
    
    return comparison_table

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
    
    # 验证2025年清明节预测
    print("开始验证2025年清明节天气预测...")
    validation_results = validate_2025_qingming(available_cities)
    
    if validation_results:
        print("\n验证完成！结果已保存为CSV文件和图表。")
    else:
        print("\n验证失败，未能获取有效的验证结果。")
    
    return validation_results

if __name__ == "__main__":
    main()

