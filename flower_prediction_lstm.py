import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# plt.rcParams
class FlowerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(FlowerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出进行预测
        out = self.fc(lstm_out[:, -1, :])
        return out

def prepare_sequence_data(weather_data, flower_data, seq_length=60):
    """
    准备LSTM的序列数据
    
    参数:
    weather_data: 气象数据DataFrame
    flower_data: 花卉数据DataFrame
    seq_length: 序列长度（使用开花前多少天的数据）
    
    返回:
    X_sequences, y_values
    """
    X_sequences = []
    y_values = []
    
    # 确保气象数据按日期排序
    weather_data = weather_data.sort_values('DATE')
    
    # 选择要使用的特征
    features = ['TEMP', 'MAX', 'MIN', 'PRCP', 'DEWP', 'WDSP']
    
    # 对特征进行归一化
    scaler = MinMaxScaler()
    weather_features = weather_data[features].values
    weather_features = scaler.fit_transform(weather_features)
    
    # 为每个开花记录创建序列
    for _, row in flower_data.iterrows():
        year = int(row['年份'])
        flowering_doy = row['始花期DOY']
        
        if pd.isna(flowering_doy):
            continue
        
        # 找到开花日期
        flowering_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(flowering_doy)-1)
        
        # 获取开花前seq_length天的数据
        seq_start_date = flowering_date - pd.Timedelta(days=seq_length)
        
        # 提取这段时间的气象数据
        mask = (weather_data['DATE'] >= seq_start_date) & (weather_data['DATE'] < flowering_date)
        seq_data = weather_data[mask]
        
        # 确保有足够的数据
        if len(seq_data) < seq_length * 0.8:  # 允许有一些缺失
            continue
            
        # 填充或截断到固定长度
        if len(seq_data) < seq_length:
            # 数据不足，在开头填充
            pad_size = seq_length - len(seq_data)
            padded_data = np.zeros((pad_size, len(features)))
            seq_features = np.vstack((padded_data, seq_data[features].values))
        else:
            # 数据足够，取最近的seq_length天
            seq_features = seq_data[features].values[-seq_length:]
        
        # 归一化
        seq_features = scaler.transform(seq_features)
        
        X_sequences.append(seq_features)
        y_values.append(flowering_doy)
    
    return np.array(X_sequences), np.array(y_values), scaler

def train_lstm_model(X_sequences, y_values, epochs=100, batch_size=4):
    """训练LSTM模型"""
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_sequences)
    y_tensor = torch.FloatTensor(y_values).view(-1, 1)
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # 80%用于训练，20%用于验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = X_sequences.shape[2]  # 特征数量
    model = FlowerLSTM(input_size)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('lstm_training_loss.png')
    
    return model

def predict_with_lstm(model, weather_data, predict_year, seq_length=60, scaler=None):
    """使用LSTM模型预测开花日期"""
    # 获取预测年份前seq_length天的数据
    # 修改：使用最近一年的同期数据代替2026年的数据
    
    # 找出数据集中最近的年份
    latest_year = weather_data['DATE'].dt.year.max()
    
    # 使用最近一年的3-4月数据作为预测基础
    predict_date = pd.Timestamp(year=latest_year, month=4, day=1)
    seq_start_date = predict_date - pd.Timedelta(days=seq_length)
    
    # 提取这段时间的气象数据
    mask = (weather_data['DATE'] >= seq_start_date) & (weather_data['DATE'] < predict_date)
    seq_data = weather_data[mask]
    
    # 确保有足够的数据
    if len(seq_data) < seq_length * 0.8:
        print(f"警告: 没有足够的气象数据用于预测，使用最近{seq_length}天的数据")
        # 使用最近的seq_length天数据
        seq_data = weather_data.sort_values('DATE').tail(seq_length)
        if len(seq_data) < seq_length * 0.8:
            return None
    
    # 准备特征
    features = ['TEMP', 'MAX', 'MIN', 'PRCP', 'DEWP', 'WDSP']
    
    # 填充或截断到固定长度
    if len(seq_data) < seq_length:
        pad_size = seq_length - len(seq_data)
        padded_data = np.zeros((pad_size, len(features)))
        seq_features = np.vstack((padded_data, seq_data[features].values))
    else:
        seq_features = seq_data[features].values[-seq_length:]
    
    # 归一化
    seq_features = scaler.transform(seq_features)
    
    # 转换为PyTorch张量并添加批次维度
    X_tensor = torch.FloatTensor(seq_features).unsqueeze(0)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred_doy = model(X_tensor).item()
    
    # 将DOY转换为日期
    pred_date = pd.Timestamp(year=predict_year, month=1, day=1) + pd.Timedelta(days=int(pred_doy)-1)
    
    return pred_date, pred_doy

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

def lstm_flower_prediction(flower_name, city_name, predict_year=2026):
    """使用LSTM预测花卉开花时间"""
    print(f"\n开始使用LSTM处理{city_name}的{flower_name}开花预测...")
    
    # 读取花卉历史开花数据
    flower_data_path = 'data2/Flower_Bloom_Data_2005_2025.csv'
    if not os.path.exists(flower_data_path):
        print(f"未找到花卉历史开花数据文件: {flower_data_path}")
        return None
    
    # 读取花卉数据（使用与原代码相同的方法）
    try:
        flower_data = pd.read_csv(flower_data_path)
    except UnicodeDecodeError:
        try:
            flower_data = pd.read_csv(flower_data_path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                flower_data = pd.read_csv(flower_data_path, encoding='gb18030')
            except Exception as e:
                print(f"无法读取花卉数据文件，错误: {e}")
                return None
    
    # 筛选特定花卉和城市的数据
    flower_city_data = flower_data[(flower_data['品类'] == flower_name) & (flower_data['地区'] == city_name)]
    if flower_city_data.empty:
        print(f"未找到{flower_name}在{city_name}的历史开花数据！")
        return None
    
    # 处理花卉数据，转换日期格式
    flower_city_data['始花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['始花期'], row['年份']), axis=1)
    flower_city_data['盛花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['盛花期'], row['年份']), axis=1)
    flower_city_data['末花期DOY'] = flower_city_data.apply(
        lambda row: convert_date_to_doy(row['末花期'], row['年份']), axis=1)
    # 计算花期长度
    flower_city_data['花期长度'] = flower_city_data.apply(
        lambda row: row['末花期DOY'] - row['始花期DOY'] if pd.notna(row['末花期DOY']) and pd.notna(row['始花期DOY']) else None, axis=1)

    
    # 读取城市气象数据文件
    files = glob.glob(f'{city_name}/{city_name}/*.csv')
    if not files:
        print(f"未找到{city_name}的气象数据文件！")
        return None
    
    # 合并所有气象数据
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, quotechar='"', low_memory=False)
            dfs.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, quotechar='"', low_memory=False, encoding='gbk')
                dfs.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    if not dfs:
        print(f"未能成功读取任何{city_name}的气象数据文件")
        return None
        
    weather_data = pd.concat(dfs, ignore_index=True)
    
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
    
    # 准备LSTM的序列数据
    X_sequences, y_values, scaler = prepare_sequence_data(weather_data, flower_city_data)
    
    if len(X_sequences) < 10:
        print(f"警告: {flower_name}在{city_name}的有效序列数据不足({len(X_sequences)}条)，LSTM预测可能不准确")
        if len(X_sequences) == 0:
            return None
    
    # 训练LSTM模型
    lstm_model = train_lstm_model(X_sequences, y_values)
    
    # 使用LSTM模型预测2026年开花日期
     # 使用LSTM模型预测2026年开花日期
    prediction_result = predict_with_lstm(lstm_model, weather_data, predict_year, scaler=scaler)
    
    # 检查预测结果是否为None
    if prediction_result is None:
        print(f"无法为{flower_name}在{city_name}的{predict_year}年开花进行预测")
        return None
        
    pred_date, pred_doy = prediction_result
    # pred_date, pred_doy = predict_with_lstm(lstm_model, weather_data, predict_year, scaler=scaler)
    
    # 评估模型性能 - 修正这里的错误，使用lstm_model而不是model
    lstm_model.eval()
    with torch.no_grad():
        y_pred = lstm_model(torch.FloatTensor(X_sequences)).numpy().flatten()
    
    mae = mean_absolute_error(y_values, y_pred)
    r2 = r2_score(y_values, y_pred)
    
    print(f"\nLSTM预测结果: {flower_name}在{city_name}的{predict_year}年开花预测:")
    print(f"预测始花期: {pred_date.strftime('%Y-%m-%d')}")
    print(f"模型平均误差: {mae:.2f}天, R²: {r2:.2f}, 样本数: {len(X_sequences)}")
    
    return {
        'flower_name': flower_name,
        'city_name': city_name,
        'predicted_flowering_date': pred_date.strftime('%Y-%m-%d'),
        'predicted_flowering_doy': int(pred_doy),
        'model_mae_days': mae,
        'model_r2': r2,
        'sample_size': len(X_sequences)
    }
    
def main():
    """主函数：预测所有花卉和城市组合"""
    # 从flower_prediction.py中获取已知有效的花卉和城市组合
    flowers_cities = [
        ('樱花', '武汉'),
        ('牡丹', '洛阳'),
        ('梅花', '杭州'),
        ('油菜花', '婺源'),
        ('杏花', '吐鲁番'),
        ('杜鹃花', '毕节')
    ]
    
    # 存储所有预测结果
    all_predictions = []
    
    # 对每个花卉和城市组合进行预测
    for flower, city in flowers_cities:
        result = lstm_flower_prediction(flower, city)
        if result:
            all_predictions.append(result)
    
    # 将结果保存为CSV
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)
        results_df.to_csv('lstm_flower_predictions_2026.csv', index=False)
        print("\n所有预测结果已保存到 lstm_flower_predictions_2026.csv")
    else:
        print("\n警告：没有成功生成任何预测结果！")
    
    return all_predictions

if __name__ == "__main__":
    main()