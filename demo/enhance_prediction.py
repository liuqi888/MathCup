import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def enhance_model_with_latest_data(city, model_dict, scaler, days_before=7):
    """
    使用最新天气实况数据更新预测模型
    
    参数:
    city: 城市名称
    model_dict: 包含各模型的字典
    scaler: 特征标准化器
    days_before: 使用清明节前多少天的最新数据
    
    返回:
    更新后的模型字典和标准化器
    """
    print(f"\n正在使用{city}最新天气实况数据更新模型...")
    
    # 获取最新的气象数据
    latest_data_files = glob.glob(f'{city}/{city}/latest/*.csv')
    if not latest_data_files:
        print(f"未找到{city}的最新气象数据，尝试使用主数据目录")
        latest_data_files = glob.glob(f'{city}/{city}/*.csv')
        
    if not latest_data_files:
        print(f"未找到{city}的气象数据，无法更新模型")
        return model_dict, scaler
    
    # 读取最新数据
    latest_dfs = []
    for file in latest_data_files:
        try:
            df = pd.read_csv(file, quotechar='"', low_memory=False)
            latest_dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    if not latest_dfs:
        return model_dict, scaler
        
    latest_data = pd.concat(latest_dfs, ignore_index=True)
    
    # 数据预处理
    latest_data['DATE'] = pd.to_datetime(latest_data['DATE'])
    latest_data['YEAR'] = latest_data['DATE'].dt.year
    latest_data['MONTH'] = latest_data['DATE'].dt.month
    latest_data['DAY'] = latest_data['DATE'].dt.day
    
    # 清理数据
    numeric_cols = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'PRCP']
    for col in numeric_cols:
        latest_data[col] = pd.to_numeric(latest_data[col].astype(str).str.strip(), errors='coerce')
    
    # 创建降雨标签
    latest_data['RAIN'] = (latest_data['PRCP'] > 0).astype(int)
    
    # 计算历史同期降雨概率
    historical_rain_prob = {}
    for month in range(1, 13):
        for day in range(1, 32):
            month_day_data = latest_data[(latest_data['MONTH'] == month) & (latest_data['DAY'] == day)]
            if len(month_day_data) > 0:
                historical_rain_prob[(month, day)] = month_day_data['RAIN'].mean()
    
    # 添加历史同期降雨概率作为特征
    latest_data['HIST_RAIN_PROB'] = latest_data.apply(
        lambda row: historical_rain_prob.get((row['MONTH'], row['DAY']), 0), 
        axis=1
    )
    
    # 添加季节性特征
    latest_data['SIN_MONTH'] = np.sin(2 * np.pi * latest_data['MONTH'] / 12)
    latest_data['COS_MONTH'] = np.cos(2 * np.pi * latest_data['MONTH'] / 12)
    
    # 筛选最近的数据
    qingming_date = pd.to_datetime('2026-04-04')  # 2026年清明节
    start_date = qingming_date - pd.Timedelta(days=days_before)
    recent_data = latest_data[latest_data['DATE'] >= start_date]
    
    if recent_data.empty:
        print(f"没有找到{city}最近{days_before}天的数据，使用最近一年的数据")
        start_date = qingming_date - pd.Timedelta(days=365)
        recent_data = latest_data[latest_data['DATE'] >= start_date]
    
    if recent_data.empty:
        print(f"没有找到{city}最近一年的数据，无法更新模型")
        return model_dict, scaler
    
    print(f"找到{len(recent_data)}条最新天气记录，正在更新模型...")
    
    # 选择特征
    features = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MONTH', 'DAY', 
                'HIST_RAIN_PROB', 'SIN_MONTH', 'COS_MONTH']
    
    # 确保所有特征都存在
    valid_features = [f for f in features if f in recent_data.columns]
    X_recent = recent_data[valid_features].dropna()
    y_recent = recent_data['RAIN'].loc[X_recent.index]
    
    if len(X_recent) < 5:
        print(f"有效数据不足，无法更新模型")
        return model_dict, scaler
    
    # 标准化特征
    X_recent_scaled = scaler.transform(X_recent)
    
    # 更新各个模型
    updated_models = {}
    
    # 更新Logistic回归模型
    if 'logistic' in model_dict:
        try:
            if hasattr(model_dict['logistic'], 'partial_fit'):
                model_dict['logistic'].partial_fit(X_recent_scaled, y_recent, classes=[0, 1])
            else:
                # 重新训练
                new_model = LogisticRegression(random_state=42, max_iter=1000)
                new_model.fit(X_recent_scaled, y_recent)
                model_dict['logistic'] = new_model
            updated_models['logistic'] = True
        except Exception as e:
            print(f"更新Logistic回归模型时出错: {e}")
    
    # 更新SVM模型 (SVM不支持增量学习，需要重新训练)
    if 'svm' in model_dict:
        try:
            new_model = SVC(random_state=42, probability=True)
            new_model.fit(X_recent_scaled, y_recent)
            model_dict['svm'] = new_model
            updated_models['svm'] = True
        except Exception as e:
            print(f"更新SVM模型时出错: {e}")
    
    # 更新随机森林模型
    if 'random_forest' in model_dict:
        try:
            # 随机森林不支持标准增量学习，但可以通过调整参数部分更新
            new_model = RandomForestClassifier(random_state=42, warm_start=True)
            new_model.fit(X_recent_scaled, y_recent)
            model_dict['random_forest'] = new_model
            updated_models['random_forest'] = True
        except Exception as e:
            print(f"更新随机森林模型时出错: {e}")
    
    # 更新梯度提升树模型
    if 'gradient_boosting' in model_dict:
        try:
            # 梯度提升树可以通过warm_start参数实现部分更新
            new_model = GradientBoostingClassifier(random_state=42, warm_start=True)
            new_model.fit(X_recent_scaled, y_recent)
            model_dict['gradient_boosting'] = new_model
            updated_models['gradient_boosting'] = True
        except Exception as e:
            print(f"更新梯度提升树模型时出错: {e}")
    
    print(f"已成功更新模型: {', '.join(updated_models.keys())}")
    
    return model_dict, scaler

def apply_model_output_statistics(predictions, validation_results=None):
    """
    应用模型输出统计(MOS)方法修正预测结果
    
    参数:
    predictions: 原始预测结果DataFrame
    validation_results: 验证结果，用于计算偏差
    
    返回:
    修正后的预测结果
    """
    print("\n应用MOS修正预测结果...")
    
    # 复制原始预测结果
    corrected_predictions = predictions.copy()
    
    # 如果提供了验证结果，计算偏差
    if validation_results is not None:
        # 计算各模型的预测偏差
        biases = {}
        
        for city_result in validation_results:
            city = city_result['city']
            preds = city_result['predictions']
            
            # 确保有实际降雨数据
            if '实际降雨' in preds.columns:
                for model in ['Logistic回归', 'SVM', '随机森林', '梯度提升树']:
                    if model in preds.columns:
                        # 计算预测偏差 (实际值 - 预测值)
                        actual = preds['实际降雨'].values
                        predicted = preds[model].values
                        bias = np.mean(actual - predicted)
                        
                        if city not in biases:
                            biases[city] = {}
                        biases[city][model] = bias
        
        # 应用偏差修正
        for city, city_biases in biases.items():
            city_preds = corrected_predictions[corrected_predictions['city'] == city]
            if not city_preds.empty:
                for model, bias in city_biases.items():
                    # 应用加法偏差修正
                    idx = corrected_predictions['city'] == city
                    corrected_predictions.loc[idx, model] = corrected_predictions.loc[idx, model] + bias
                    # 确保预测值为0或1
                    corrected_predictions.loc[idx, model] = (corrected_predictions.loc[idx, model] >= 0.5).astype(int)
    else:
        # 如果没有验证结果，使用保守的偏差修正
        # 对于南方城市，略微提高降雨概率
        south_cities = ['婺源', '毕节']
        for city in south_cities:
            idx = corrected_predictions['city'] == city
            if any(idx):
                # 提高降雨概率
                for model in ['Logistic回归', 'SVM', '随机森林', '梯度提升树']:
                    if model in corrected_predictions.columns:
                        # 增加0.1的偏差，但确保不超过1
                        corrected_predictions.loc[idx, f'{model}概率'] = (
                            corrected_predictions.loc[idx, f'{model}概率'] + 0.1
                        ).clip(0, 1)
                
                # 重新计算加权概率
                if all(col in corrected_predictions.columns for col in 
                      ['Logistic概率', 'SVM概率', '随机森林概率', '梯度提升树概率']):
                    corrected_predictions.loc[idx, '加权概率'] = (
                        corrected_predictions.loc[idx, 'Logistic概率'] * 0.25 + 
                        corrected_predictions.loc[idx, 'SVM概率'] * 0.25 + 
                        corrected_predictions.loc[idx, '随机森林概率'] * 0.25 +
                        corrected_predictions.loc[idx, '梯度提升树概率'] * 0.25
                    )
                    
                    # 更新加权投票
                    corrected_predictions.loc[idx, '加权投票'] = (
                        corrected_predictions.loc[idx, '加权概率'] >= 0.5
                    ).astype(int)
    
    print("MOS修正已应用")
    return corrected_predictions

def integrate_nwp_data(predictions, nwp_data=None):
    """
    整合数值天气预报(NWP)数据
    
    参数:
    predictions: 原始预测结果
    nwp_data: 数值天气预报数据，如果为None则尝试加载
    
    返回:
    整合NWP后的预测结果
    """
    print("\n整合数值天气预报数据...")
    
    # 复制原始预测结果
    integrated_predictions = predictions.copy()
    
    # 如果没有提供NWP数据，创建模拟数据（实际应用中应加载真实NWP数据）
    if nwp_data is None:
        print("未提供NWP数据，创建模拟数据用于演示")
        
        # 创建一个包含所有城市和日期的DataFrame
        cities = integrated_predictions['city'].unique()
        dates = integrated_predictions['DATE'].unique()
        
        nwp_rows = []
        for city in cities:
            for date in dates:
                # 为南方城市设置较高的降雨概率
                if city in ['婺源', '毕节']:
                    rain_prob = np.random.uniform(0.4, 0.7)
                else:
                    rain_prob = np.random.uniform(0.1, 0.4)
                
                nwp_rows.append({
                    'city': city,
                    'DATE': date,
                    'NWP_RAIN_PROB': rain_prob
                })
        
        nwp_data = pd.DataFrame(nwp_rows)
    
    # 确保NWP数据包含必要的列
    required_cols = ['city', 'DATE', 'NWP_RAIN_PROB']
    if not all(col in nwp_data.columns for col in required_cols):
        print("NWP数据缺少必要的列，无法整合")
        return integrated_predictions
    
    # 将NWP数据与预测结果合并
    integrated_predictions = pd.merge(
        integrated_predictions, 
        nwp_data[required_cols],
        on=['city', 'DATE'], 
        how='left'
    )
    
    # 如果有缺失的NWP数据，使用原始预测
    if integrated_predictions['NWP_RAIN_PROB'].isna().any():
        print("警告: 部分日期缺少NWP数据，将使用原始预测")
        integrated_predictions['NWP_RAIN_PROB'] = integrated_predictions['NWP_RAIN_PROB'].fillna(
            integrated_predictions['加权概率']
        )
    
    # 整合ML模型预测和NWP预测
    # 这里使用简单的加权平均，可以根据需要调整权重
    ml_weight = 0.7  # 机器学习模型权重
    nwp_weight = 0.3  # NWP模型权重
    
    integrated_predictions['整合概率'] = (
        integrated_predictions['加权概率'] * ml_weight + 
        integrated_predictions['NWP_RAIN_PROB'] * nwp_weight
    )
    
    # 更新预测结果
    integrated_predictions['整合预测'] = (integrated_predictions['整合概率'] >= 0.5).astype(int)
    
    print("NWP数据整合完成")
    return integrated_predictions

def enhance_qingming_prediction(validation_results=None):
    """
    增强2026年清明节降雨预测
    
    参数:
    validation_results: 2025年验证结果，用于MOS修正
    
    返回:
    增强后的预测结果
    """
    from rain_prediction import predict_rainfall  # 导入原始预测函数
    
    print("\n开始增强2026年清明节降雨预测...")
    
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
    all_predictions = []
    
    # 对每个城市进行预测
    for city in available_cities:
        # 1. 获取基础预测结果
        base_result = predict_rainfall(city)
        if not base_result:
            print(f"无法获取{city}的基础预测结果，跳过")
            continue
        
        # 2. 使用最新数据更新模型
        models = {
            'logistic': base_result.get('models', {}).get('logistic'),
            'svm': base_result.get('models', {}).get('svm'),
            'random_forest': base_result.get('models', {}).get('random_forest'),
            'gradient_boosting': base_result.get('models', {}).get('gradient_boosting')
        }
        
        scaler = base_result.get('scaler')
        
        if all(model is not None for model in models.values()) and scaler is not None:
            models, scaler = enhance_model_with_latest_data(city, models, scaler)
        
        # 添加城市列
        predictions = base_result['predictions']
        predictions['city'] = city
        all_predictions.append(predictions)
        
        # 保存结果
        all_results.append({
            'city': city,
            'base_result': base_result,
            'updated_models': models,
            'scaler': scaler
        })
    
    if not all_predictions:
        print("没有获取到任何预测结果，无法继续")
        return None
    
    # 合并所有城市的预测
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # 3. 应用MOS修正
    corrected_predictions = apply_model_output_statistics(combined_predictions, validation_results)
    
    # 4. 整合数值天气预报
    enhanced_predictions = integrate_nwp_data(corrected_predictions)
    
    # 5. 创建最终预测表格
    final_predictions = pd.DataFrame()
    
    for city in available_cities:
        city_preds = enhanced_predictions[enhanced_predictions['city'] == city]
        
        for day in range(4, 7):  # 4月4-6日
            day_preds = city_preds[city_preds['DATE'].dt.day == day]
            if not day_preds.empty:
                # 使用整合预测结果
                if '整合预测' in day_preds.columns:
                    pred = day_preds['整合预测'].iloc[0]
                    prob = day_preds['整合概率'].iloc[0]
                else:
                    pred = day_preds['加权投票'].iloc[0]
                    prob = day_preds['加权概率'].iloc[0]
                
                final_predictions.loc[city, f'4月{day}日'] = '有雨' if pred == 1 else '无雨'
                final_predictions.loc[city, f'4月{day}日概率'] = f"{prob:.2f}"
    
    print("\n2026年清明节增强降雨预测结果:")
    print(final_predictions)
    
    # 保存结果
    final_predictions.to_csv('2026年清明节增强降雨预测.csv', encoding='utf-8-sig')
    
    # 创建可视化
    create_enhanced_prediction_visualizations(final_predictions, available_cities)
    
    return {
        'final_predictions': final_predictions,
        'enhanced_predictions': enhanced_predictions,
        'all_results': all_results
    }

def create_enhanced_prediction_visualizations(final_predictions, cities):
    """
    为增强预测创建可视化
    
    参数:
    final_predictions: 最终预测结果
    cities: 城市列表
    """
    # 1. 创建降雨概率热力图
    plt.figure(figsize=(12, 8))
    
    # 提取概率值并转换为浮点数
    heatmap_data = pd.DataFrame(index=cities, columns=['4月4日', '4月5日', '4月6日'])
    
    for city in cities:
        for day in range(4, 7):
            col_name = f'4月{day}日概率'
            if col_name in final_predictions.columns and city in final_predictions.index:
                # 从字符串中提取概率值
                prob_str = final_predictions.loc[city, col_name]
                try:
                    prob = float(prob_str)
                    heatmap_data.loc[city, f'4月{day}日'] = prob
                except:
                    heatmap_data.loc[city, f'4月{day}日'] = np.nan
    
    # 确保数据是浮点型
    heatmap_data = heatmap_data.astype(float)
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.2f', 
                linewidths=.5, cbar_kws={'label': '降雨概率'})
    plt.title('2026年清明节各城市降雨概率预测')
    plt.tight_layout()
    plt.savefig('2026年清明节降雨概率热力图.png')
    plt.close()
    
    # 2. 创建降雨预测条形图
    plt.figure(figsize=(14, 8))
    
    # 准备数据
    rain_data = pd.DataFrame(index=cities, columns=['4月4日', '4月5日', '4月6日'])
    
    for city in cities:
        for day in range(4, 7):
            col_name = f'4月{day}日'
            if col_name in final_predictions.columns and city in final_predictions.index:
                # 将"有雨"/"无雨"转换为1/0
                prediction = final_predictions.loc[city, col_name]
                rain_data.loc[city, col_name] = 1 if prediction == '有雨' else 0
    
    # 转置数据以便于绘图
    rain_data_t = rain_data.T
    
    # 绘制条形图
    ax = rain_data_t.plot(kind='bar', figsize=(14, 8), width=0.7)
    
    # 设置图表属性
    plt.title('2026年清明节各城市降雨预测')
    plt.xlabel('日期')
    plt.ylabel('预测结果')
    plt.yticks([0, 1], ['无雨', '有雨'])
    plt.legend(title='城市')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')
    
    plt.tight_layout()
    plt.savefig('2026年清明节降雨预测条形图.png')
    plt.close()
    
    # 3. 创建城市分组比较图
    plt.figure(figsize=(15, 10))
    
    # 按地理位置分组城市
    north_cities = [city for city in cities if city in ['西安', '洛阳', '吐鲁番']]
    central_cities = [city for city in cities if city in ['武汉', '杭州']]
    south_cities = [city for city in cities if city in ['婺源', '毕节']]
    
    # 计算各组的平均降雨概率
    group_data = {
        '北方城市': {},
        '中部城市': {},
        '南方城市': {}
    }
    
    for day in range(4, 7):
        col_name = f'4月{day}日概率'
        day_label = f'4月{day}日'
        
        # 计算北方城市平均值
        north_probs = []
        for city in north_cities:
            if col_name in final_predictions.columns and city in final_predictions.index:
                try:
                    prob = float(final_predictions.loc[city, col_name])
                    north_probs.append(prob)
                except:
                    pass
        if north_probs:
            group_data['北方城市'][day_label] = np.mean(north_probs)
        
        # 计算中部城市平均值
        central_probs = []
        for city in central_cities:
            if col_name in final_predictions.columns and city in final_predictions.index:
                try:
                    prob = float(final_predictions.loc[city, col_name])
                    central_probs.append(prob)
                except:
                    pass
        if central_probs:
            group_data['中部城市'][day_label] = np.mean(central_probs)
        
        # 计算南方城市平均值
        south_probs = []
        for city in south_cities:
            if col_name in final_predictions.columns and city in final_predictions.index:
                try:
                    prob = float(final_predictions.loc[city, col_name])
                    south_probs.append(prob)
                except:
                    pass
        if south_probs:
            group_data['南方城市'][day_label] = np.mean(south_probs)
    
    # 创建DataFrame
    group_df = pd.DataFrame(group_data)
    
    # 绘制分组条形图
    ax = group_df.plot(kind='bar', figsize=(15, 10), width=0.7)
    
    plt.title('2026年清明节不同地区降雨概率比较')
    plt.xlabel('日期')
    plt.ylabel('平均降雨概率')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('2026年清明节地区降雨概率比较.png')
    plt.close()
    
    print("可视化图表已生成")

def create_rolling_forecast_visualization(city, predictions_history):
    """
    创建滚动预测可视化，展示预测如何随时间变化
    
    参数:
    city: 城市名称
    predictions_history: 包含历史预测的字典，键为预测日期，值为预测结果
    """
    if not predictions_history:
        print(f"没有{city}的历史预测数据，无法创建滚动预测可视化")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    dates = sorted(predictions_history.keys())
    
    # 提取4月4-6日的预测概率
    probs_4th = []
    probs_5th = []
    probs_6th = []
    
    for date in dates:
        preds = predictions_history[date]
        if '4月4日概率' in preds:
            try:
                probs_4th.append(float(preds['4月4日概率']))
            except:
                probs_4th.append(np.nan)
        else:
            probs_4th.append(np.nan)
            
        if '4月5日概率' in preds:
            try:
                probs_5th.append(float(preds['4月5日概率']))
            except:
                probs_5th.append(np.nan)
        else:
            probs_5th.append(np.nan)
            
        if '4月6日概率' in preds:
            try:
                probs_6th.append(float(preds['4月6日概率']))
            except:
                probs_6th.append(np.nan)
        else:
            probs_6th.append(np.nan)
    
    # 绘制趋势线
    plt.plot(dates, probs_4th, 'o-', label='4月4日')
    plt.plot(dates, probs_5th, 's-', label='4月5日')
    plt.plot(dates, probs_6th, '^-', label='4月6日')
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('预测日期')
    plt.ylabel('降雨概率')
    plt.title(f'{city}清明节降雨预测概率随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{city}_滚动预测趋势.png')
    plt.close()
    
    print(f"{city}滚动预测可视化已生成")

def perform_rolling_forecast(city, days_before=10):
    """
    执行滚动预测，模拟从指定天数前开始每天进行一次预测
    
    参数:
    city: 城市名称
    days_before: 从清明节前多少天开始预测
    
    返回:
    滚动预测结果
    """
    from rain_prediction import predict_rainfall
    
    print(f"\n开始{city}的滚动预测模拟...")
    
    # 清明节日期
    qingming_date = pd.to_datetime('2026-04-04')
    
    # 存储每天的预测结果
    predictions_history = {}
    
    # 从指定天数前开始，每天进行一次预测
    for day in range(days_before, 0, -1):
        current_date = qingming_date - pd.Timedelta(days=day)
        print(f"\n模拟在{current_date.strftime('%Y-%m-%d')}进行预测...")
        
        # 获取基础预测结果
        base_result = predict_rainfall(city)
        if not base_result:
            print(f"无法获取{city}的基础预测结果，跳过")
            continue
        
        # 提取预测结果
        predictions = base_result['predictions']
        
        # 创建当天的预测结果摘要
        day_prediction = {}
        
        for pred_day in range(4, 7):  # 4月4-6日
            day_preds = predictions[predictions['DATE'].dt.day == pred_day]
            if not day_preds.empty:
                # 使用加权预测结果
                pred = day_preds['加权投票'].iloc[0]
                prob = day_preds['加权概率'].iloc[0]
                
                day_prediction[f'4月{pred_day}日'] = '有雨' if pred == 1 else '无雨'
                day_prediction[f'4月{pred_day}日概率'] = f"{prob:.2f}"
        
        # 存储当天的预测结果
        predictions_history[current_date] = day_prediction
    
    # 创建滚动预测可视化
    create_rolling_forecast_visualization(city, predictions_history)
    
    return predictions_history

def main():
    """
    主函数
    """
    # 从验证模块导入验证结果
    try:
        from validate_2025_qingming import main as validate_main
        validation_results = validate_main()
    except ImportError:
        print("无法导入验证模块，将不使用验证结果进行MOS修正")
        validation_results = None
    
    # 增强2026年清明节降雨预测
    enhanced_results = enhance_qingming_prediction(validation_results)
    
    if enhanced_results:
        print("\n增强预测完成！结果已保存为CSV文件和图表。")
        
        # 对每个城市执行滚动预测
        cities = enhanced_results['final_predictions'].index.tolist()
        for city in cities:
            perform_rolling_forecast(city, days_before=7)
    else:
        print("\n增强预测失败，未能获取有效的预测结果。")
    
    return enhanced_results

if __name__ == "__main__":
    main()





