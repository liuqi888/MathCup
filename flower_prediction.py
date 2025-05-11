import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def convert_date_to_doy(date_str, year):
    """
    将MM.DD格式的日期转换为一年中的第几天
    """
    if pd.isna(date_str):
        return None
    
    try:
        month, day = map(int, str(date_str).split('.'))
        date = pd.Timestamp(year=int(year), month=month, day=day)
        return date.dayofyear
    except Exception as e:
        print(f"日期转换错误: {date_str}, {year}, 错误: {e}")
        return None

def predict_flowering(flower_name, city_name, predict_year=2026):
    """
    根据历史气象数据预测花卉开花时间
    
    参数:
    flower_name: 花卉名称
    city_name: 城市名称
    predict_year: 预测年份
    
    返回:
    预测的开花日期和花期长度
    """
    print(f"\n开始处理{city_name}的{flower_name}开花预测...")
    
    # 读取花卉历史开花数据
    flower_data_path = 'data2/Flower_Bloom_Data_2005_2025.csv'
    if not os.path.exists(flower_data_path):
        print(f"未找到花卉历史开花数据文件: {flower_data_path}")
        return None
    
    # 读取花卉数据
    try:
        # 尝试使用UTF-8编码读取
        flower_data = pd.read_csv(flower_data_path)
    except UnicodeDecodeError:
        try:
            # 尝试使用GBK编码读取
            flower_data = pd.read_csv(flower_data_path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                # 尝试使用GB18030编码读取
                flower_data = pd.read_csv(flower_data_path, encoding='gb18030')
            except Exception as e:
                print(f"无法读取花卉数据文件，错误: {e}")
                return None
    
    print(f"读取到{len(flower_data)}条花卉记录")
    
    # 筛选特定花卉和城市的数据
    flower_city_data = flower_data[(flower_data['品类'] == flower_name) & (flower_data['地区'] == city_name)]
    if flower_city_data.empty:
        print(f"未找到{flower_name}在{city_name}的历史开花数据！")
        return None
    
    print(f"找到{len(flower_city_data)}条{flower_name}在{city_name}的历史开花记录")
    
    # 处理花卉数据，转换日期格式
    flower_city_data['始花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['始花期'], row['年份']), axis=1)
    flower_city_data['盛花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['盛花期'], row['年份']), axis=1)
    flower_city_data['末花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['末花期'], row['年份']), axis=1)
    
    # 读取城市气象数据文件
    files = glob.glob(f'{city_name}/{city_name}/*.csv')
    if not files:
        print(f"未找到{city_name}的气象数据文件！")
        return None
    
    # 合并所有气象数据
    dfs = []
    for file in files:
        try:
            # 尝试使用默认编码读取
            df = pd.read_csv(file, quotechar='"', low_memory=False)
            dfs.append(df)
        except UnicodeDecodeError:
            try:
                # 尝试使用GBK编码读取
                df = pd.read_csv(file, quotechar='"', low_memory=False, encoding='gbk')
                dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    if not dfs:
        print(f"未能成功读取任何{city_name}的气象数据文件")
        return None
        
    weather_data = pd.concat(dfs, ignore_index=True)
    print(f"读取到{len(weather_data)}条气象记录")
    
    # 数据清洗和预处理
    weather_data['DATE'] = pd.to_datetime(weather_data['DATE'])
    weather_data['YEAR'] = weather_data['DATE'].dt.year
    weather_data['MONTH'] = weather_data['DATE'].dt.month
    weather_data['DAY'] = weather_data['DATE'].dt.day
    weather_data['DOY'] = weather_data['DATE'].dt.dayofyear
    
    # 清理数据中的非数值字符
    numeric_cols = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MAX', 'MIN', 'PRCP']
    for col in numeric_cols:
        if col in weather_data.columns:
            weather_data[col] = pd.to_numeric(weather_data[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    
    # 合并花卉数据和气象数据
    merged_data = []
    
    for _, row in flower_city_data.iterrows():
        year = int(row['年份'])
        flowering_doy = row['始花期DOY']
        
        if pd.isna(flowering_doy):
            print(f"警告: {year}年的开花日期数据缺失，跳过")
            continue
        
        # 获取当年冬季和春季的气象数据（前一年11月到当年开花前）
        winter_spring_data = weather_data[
            ((weather_data['YEAR'] == year-1) & (weather_data['MONTH'] >= 11)) |
            ((weather_data['YEAR'] == year) & (weather_data['DOY'] < flowering_doy))
        ]
        
        if winter_spring_data.empty:
            print(f"警告: {year}年没有足够的冬季和春季气象数据")
            continue
        
        # 计算关键气象指标
        # 1. 冬季平均温度（前一年11-12月）
        winter_data = winter_spring_data[
            (winter_spring_data['YEAR'] == year-1) & 
            (winter_spring_data['MONTH'] >= 11)
        ]
        winter_temp = winter_data['TEMP'].mean() if not winter_data.empty else np.nan
        
        # 2. 春季平均温度（1月到开花前）
        spring_data = winter_spring_data[
            (winter_spring_data['YEAR'] == year) & 
            (winter_spring_data['MONTH'] >= 1)
        ]
        
        if spring_data.empty:
            print(f"警告: {year}年没有春季气象数据")
            continue
            
        spring_temp = spring_data['TEMP'].mean()
        
        # 3. 累积温度（Growing Degree Days）
        # 假设基础温度为5°C (41°F)
        base_temp = 41.0  # 华氏温度
        gdd_data = winter_spring_data[winter_spring_data['TEMP'] > base_temp]
        gdd = (gdd_data['TEMP'] - base_temp).sum() if not gdd_data.empty else 0
        
        # 4. 春季降水量
        spring_precip = spring_data['PRCP'].sum() if 'PRCP' in spring_data.columns else 0
        
        # 5. 冬季低温日数（低于0°C的天数）
        cold_days = len(winter_spring_data[winter_spring_data['MIN'] < 32.0]) if 'MIN' in winter_spring_data.columns else 0
        
        # 6. 日照时数（如果有数据）
        sunshine_hours = winter_spring_data['SUNSHINE'].sum() if 'SUNSHINE' in winter_spring_data.columns else 0
        
        # 7. 春季温度波动（标准差）
        spring_temp_std = spring_data['TEMP'].std() if len(spring_data) > 1 else 0
        
        # 8. 春季最高温度
        spring_max_temp = spring_data['MAX'].max() if 'MAX' in spring_data.columns else spring_data['TEMP'].max()
        
        # 9. 春季最低温度
        spring_min_temp = spring_data['MIN'].min() if 'MIN' in spring_data.columns else spring_data['TEMP'].min()
        
        # 10. 春季温度范围
        spring_temp_range = spring_max_temp - spring_min_temp
        
        # 将数据添加到列表
        data_dict = {
            'YEAR': year,
            'FLOWERING_DOY': flowering_doy,
            'FULL_BLOOM_DOY': row['盛花期DOY'],
            'END_BLOOM_DOY': row['末花期DOY'],
            'BLOOM_DURATION': row['总花期长度'],
            'WINTER_TEMP': winter_temp,
            'SPRING_TEMP': spring_temp,
            'GDD': gdd,
            'SPRING_PRECIP': spring_precip,
            'COLD_DAYS': cold_days,
            'SUNSHINE_HOURS': sunshine_hours,
            'SPRING_TEMP_STD': spring_temp_std,
            'SPRING_MAX_TEMP': spring_max_temp,
            'SPRING_MIN_TEMP': spring_min_temp,
            'SPRING_TEMP_RANGE': spring_temp_range
        }
        merged_data.append(data_dict)
    
    # 创建DataFrame
    model_data = pd.DataFrame(merged_data)
    
    # 处理缺失值
    model_data = model_data.dropna(subset=['FLOWERING_DOY'])
    
    # 检查样本数量
    if len(model_data) < 5:
        print(f"警告: {flower_name}在{city_name}的有效数据样本不足({len(model_data)}条)，预测可能不准确")
        if len(model_data) == 0:
            return None
    
    # 选择可用的特征
    all_features = ['WINTER_TEMP', 'SPRING_TEMP', 'GDD', 'SPRING_PRECIP', 'COLD_DAYS', 
                   'SUNSHINE_HOURS', 'SPRING_TEMP_STD', 'SPRING_MAX_TEMP', 'SPRING_MIN_TEMP', 
                   'SPRING_TEMP_RANGE']
    
    # 移除缺失值过多的特征
    valid_features = []
    for feature in all_features:
        if feature in model_data.columns and model_data[feature].notna().sum() > len(model_data) * 0.7:
            valid_features.append(feature)
    
    print(f"使用的特征: {valid_features}")
    
    # 填充剩余的缺失值
    for feature in valid_features:
        model_data[feature] = model_data[feature].fillna(model_data[feature].median())
    
    # 特征和目标变量
    X = model_data[valid_features]
    y_flowering = model_data['FLOWERING_DOY']  # 始花期
    y_full_bloom = model_data['FULL_BLOOM_DOY']  # 盛花期
    y_end_bloom = model_data['END_BLOOM_DOY']  # 末花期
    y_duration = model_data['BLOOM_DURATION']  # 花期长度
    
    # 打印特征相关性
    print("\n特征相关性分析:")
    for feature in valid_features:
        correlation = np.corrcoef(X[feature], y_flowering)[0, 1]
        print(f"{feature} 与始花期的相关性: {correlation:.4f}")
    
    # 训练始花期预测模型
    rf_model_flowering = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_flowering.fit(X, y_flowering)
    
    # 训练盛花期预测模型
    rf_model_full_bloom = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_full_bloom.fit(X, y_full_bloom)
    
    # 训练末花期预测模型
    rf_model_end_bloom = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_end_bloom.fit(X, y_end_bloom)
    
    # 训练花期长度预测模型
    rf_model_duration = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_duration.fit(X, y_duration)
    
    # 打印特征重要性
    print("\n特征重要性:")
    for feature, importance in zip(valid_features, rf_model_flowering.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    # 为预测年份创建预测数据
    # 使用历史数据的平均值作为预测特征
    pred_features = {}
    
    # 计算各特征的预测值
    for feature in valid_features:
        if feature == 'WINTER_TEMP':
            # 冬季平均温度（使用历史11-12月数据）
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([11, 12])]['TEMP'].mean()
        elif feature == 'SPRING_TEMP':
            # 春季平均温度（使用历史1-4月数据）
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].mean()
        elif feature == 'GDD':
            # 累积温度（使用历史平均值）
            pred_features[feature] = model_data['GDD'].mean()
        elif feature == 'SPRING_PRECIP':
            # 春季降水量（使用历史1-4月数据）
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['PRCP'].sum() / \
                                    len(weather_data['YEAR'].unique())  # 平均每年的总降水量
        elif feature == 'COLD_DAYS':
            # 冬季低温日数（使用历史平均值）
            pred_features[feature] = model_data['COLD_DAYS'].mean()
        elif feature == 'SPRING_TEMP_STD':
            # 春季温度波动
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].std()
        elif feature == 'SPRING_MAX_TEMP':
            # 春季最高温度
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['MAX'].max() \
                if 'MAX' in weather_data.columns else weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].max()
        elif feature == 'SPRING_MIN_TEMP':
            # 春季最低温度
            pred_features[feature] = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['MIN'].min() \
                if 'MIN' in weather_data.columns else weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].min()
        elif feature == 'SPRING_TEMP_RANGE':
            # 春季温度范围
            max_temp = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['MAX'].max() \
                if 'MAX' in weather_data.columns else weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].max()
            min_temp = weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['MIN'].min() \
                if 'MIN' in weather_data.columns else weather_data[weather_data['MONTH'].isin([1, 2, 3, 4])]['TEMP'].min()
            pred_features[feature] = max_temp - min_temp
        elif feature == 'SUNSHINE_HOURS':
            # 日照时数
            pred_features[feature] = model_data['SUNSHINE_HOURS'].mean() if 'SUNSHINE_HOURS' in model_data.columns else 0
    
    # 创建预测特征DataFrame
    pred_df = pd.DataFrame({k: [v] for k, v in pred_features.items()})
    
    # 预测开花日期和花期长度
    pred_flowering_doy = rf_model_flowering.predict(pred_df)[0]
    pred_full_bloom_doy = rf_model_full_bloom.predict(pred_df)[0]
    pred_end_bloom_doy = rf_model_end_bloom.predict(pred_df)[0]
    pred_duration = rf_model_duration.predict(pred_df)[0]
    
    # 将DOY转换为日期
    pred_flowering_date = pd.Timestamp(year=predict_year, month=1, day=1) + pd.Timedelta(days=int(pred_flowering_doy)-1)
    pred_full_bloom_date = pd.Timestamp(year=predict_year, month=1, day=1) + pd.Timedelta(days=int(pred_full_bloom_doy)-1)
    pred_end_bloom_date = pd.Timestamp(year=predict_year, month=1, day=1) + pd.Timedelta(days=int(pred_end_bloom_doy)-1)
    
    # 计算模型评估指标
    y_pred_flowering = rf_model_flowering.predict(X)
    mae_flowering = mean_absolute_error(y_flowering, y_pred_flowering)
    r2_flowering = r2_score(y_flowering, y_pred_flowering)
    
    # 输出结果
    result = {
        'flower_name': flower_name,
        'city_name': city_name,
        'predicted_flowering_date': pred_flowering_date.strftime('%Y-%m-%d'),
        'predicted_full_bloom_date': pred_full_bloom_date.strftime('%Y-%m-%d'),
        'predicted_end_bloom_date': pred_end_bloom_date.strftime('%Y-%m-%d'),
        'predicted_flowering_doy': int(pred_flowering_doy),
        'predicted_full_bloom_doy': int(pred_full_bloom_doy),
        'predicted_end_bloom_doy': int(pred_end_bloom_doy),
        'predicted_duration': int(pred_duration),
        'model_mae_days': mae_flowering,
        'model_r2': r2_flowering,
        'sample_size': len(model_data)
    }
    
    print(f"\n预测结果: {flower_name}在{city_name}的{predict_year}年开花预测:")
    print(f"始花期: {pred_flowering_date.strftime('%Y-%m-%d')}")
    print(f"盛花期: {pred_full_bloom_date.strftime('%Y-%m-%d')}")
    print(f"末花期: {pred_end_bloom_date.strftime('%Y-%m-%d')}")
    print(f"花期长度: 约{int(pred_duration)}天")
    print(f"模型平均误差: {mae_flowering:.2f}天, R²: {r2_flowering:.2f}, 样本数: {len(model_data)}")
    
    return result

def auto_detect_flowers_cities():
    """自动检测花卉和城市组合"""
    flower_data_path = 'data2/Flower_Bloom_Data_2005_2025.csv'
    if not os.path.exists(flower_data_path):
        print(f"未找到花卉数据文件: {flower_data_path}")
        return []
    
    try:
        # 尝试使用UTF-8编码读取
        flower_data = pd.read_csv(flower_data_path)
    except UnicodeDecodeError:
        try:
            # 尝试使用GBK编码读取（常用于中文Windows系统）
            flower_data = pd.read_csv(flower_data_path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                # 尝试使用GB18030编码读取（GBK的超集，支持更多中文字符）
                flower_data = pd.read_csv(flower_data_path, encoding='gb18030')
            except Exception as e:
                print(f"无法读取花卉数据文件，错误: {e}")
                return []
    
    unique_flowers = flower_data['品类'].unique()
    unique_cities = flower_data['地区'].unique()
    
    # 只返回在数据中实际存在的花卉和城市组合
    flowers_cities = []
    for flower in unique_flowers:
        for city in unique_cities:
            if not flower_data[(flower_data['品类'] == flower) & (flower_data['地区'] == city)].empty:
                flowers_cities.append((flower, city))
    
    return flowers_cities

def main():
    # 自动检测花卉和城市组合
    flowers_cities = auto_detect_flowers_cities()
    
    if not flowers_cities:
        print("未找到任何花卉和城市组合，请检查data2目录中的Flower_Bloom_Data_2005_2025.csv文件")
        # 手动指定一些组合用于测试
        flowers_cities = [
            ('樱花', '武汉'),
            ('牡丹', '洛阳'),
            ('梅花', '杭州'),
            ('油菜花', '婺源')
        ]
    
    print(f"检测到{len(flowers_cities)}个花卉和城市组合:")
    for flower, city in flowers_cities:
        print(f"- {flower} in {city}")
    
    results = []
    for flower, city in flowers_cities:
        result = predict_flowering(flower, city)
        if result:
            results.append(result)
    
    if not results:
        print("未能获取任何有效的预测结果！")
        return
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 按开花日期排序
    results_df = results_df.sort_values('predicted_flowering_doy')
    
    # 保存结果
    results_df.to_csv('2026年花卉开花预测.csv', encoding='utf-8-sig', index=False)
    print(f"\n预测结果已保存至 '2026年花卉开花预测.csv'")
    
    # 可视化结果 - 按城市分组
    plt.figure(figsize=(14, 10))
    cities = results_df['city_name'].unique()
    
    for i, city in enumerate(cities):
        city_data = results_df[results_df['city_name'] == city]
        plt.subplot(len(cities), 1, i+1)
        
        for _, row in city_data.iterrows():
            start_date = pd.to_datetime(row['predicted_flowering_date'])
            end_date = pd.to_datetime(row['predicted_end_bloom_date'])
            
            plt.barh(row['flower_name'], width=(end_date - start_date).days, 
                    left=start_date, height=0.5, 
                    color=plt.cm.Paired(i % 10))
            
            # 添加日期标签
            plt.text(start_date, row['flower_name'], 
                    start_date.strftime('%m-%d'), 
                    va='center', ha='right', fontsize=8)
            
        plt.title(f'{city} 2026年花卉开花预测')
        plt.xlabel('日期')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 设置x轴范围为1月到6月
        plt.xlim(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30'))
        
        # 设置x轴刻度为每月1日
        months = pd.date_range(start='2026-01-01', end='2026-06-01', freq='MS')
        plt.xticks(months, [d.strftime('%m-%d') for d in months])
    
    plt.tight_layout()
    plt.savefig('2026年花卉开花预测.png', dpi=300)
    print(f"预测可视化已保存至 '2026年花卉开花预测.png'")
    
    # 可视化结果 - 按花期时间轴
    plt.figure(figsize=(15, 8))
    
    # 创建一个包含所有花卉的时间轴
    for i, row in results_df.iterrows():
        start_date = pd.to_datetime(row['predicted_flowering_date'])
        end_date = pd.to_datetime(row['predicted_end_bloom_date'])
        flower_city = f"{row['flower_name']}({row['city_name']})"
        
        plt.barh(flower_city, width=(end_date - start_date).days, 
                left=start_date, height=0.5,
                color=plt.cm.tab20(i % 20))
        
        # 添加日期标签
        plt.text(start_date, flower_city, 
                start_date.strftime('%m-%d'), 
                va='center', ha='right', fontsize=8)
    
    plt.title('2026年各地花卉开花时间预测')
    plt.xlabel('日期')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 设置x轴范围为1月到6月
    plt.xlim(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-30'))
    
    # 设置x轴刻度为每月1日
    months = pd.date_range(start='2026-01-01', end='2026-06-01', freq='MS')
    plt.xticks(months, [d.strftime('%m-%d') for d in months])
    
    plt.tight_layout()
    plt.savefig('2026年各地花卉开花时间轴.png', dpi=300)
    print(f"花期时间轴可视化已保存至 '2026年各地花卉开花时间轴.png'")

if __name__ == "__main__":
    main()




