import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体，确保中文能正常显示
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")  # Windows系统
except:
    font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")  # macOS系统


class GrayCorrelationModel:
    def __init__(self, data, reference_column):
        """
        初始化灰色关联模型

        参数:
        data (DataFrame): 包含所有变量的数据框
        reference_column (str): 参考序列的列名
        """
        self.data = data
        self.reference_column = reference_column
        self.processed_data = None
        self.gray_relational_degrees = None

    def preprocess_data(self, method='initial'):
        """
        数据预处理 - 无量纲化

        参数:
        method (str): 无量纲化方法，'initial'为初值化，'mean'为均值化

        返回:
        DataFrame: 预处理后的数据
        """
        processed_data = pd.DataFrame(index=self.data.index)

        for column in self.data.columns:
            if method == 'initial':
                # 初值化：每个数据除以该列第一个值
                processed_data[column] = self.data[column] / self.data[column].iloc[0]
            elif method == 'mean':
                # 均值化：每个数据除以该列平均值
                processed_data[column] = self.data[column] / self.data[column].mean()
            else:
                raise ValueError("不支持的无量纲化方法")

        self.processed_data = processed_data
        return processed_data

    def calculate_gray_correlation(self, rho=0.5):
        """
        计算灰色关联度

        参数:
        rho (float): 分辨系数，取值范围(0,1)，通常取0.5

        返回:
        Series: 各比较序列与参考序列的灰色关联度
        """
        if self.processed_data is None:
            self.preprocess_data()

        # 获取参考序列
        reference = self.processed_data[self.reference_column]

        # 初始化关联度结果
        gray_relational_degrees = pd.Series(index=self.processed_data.columns, dtype=float)

        # 计算每个比较序列与参考序列的关联度
        for column in self.processed_data.columns:
            if column == self.reference_column:
                gray_relational_degrees[column] = 1.0  # 自身与自身的关联度为1
                continue

            # 计算绝对差序列
            diff = np.abs(self.processed_data[column] - reference)

            # 计算两极最大差和最小差
            min_diff = diff.min()
            max_diff = diff.max()

            # 计算关联系数
            correlation_coefficient = (min_diff + rho * max_diff) / (diff + rho * max_diff)

            # 计算关联度（关联系数的平均值）
            gray_relational_degrees[column] = correlation_coefficient.mean()

        self.gray_relational_degrees = gray_relational_degrees
        return gray_relational_degrees

    def plot_correlation_results(self):
        """
        可视化关联度结果
        """
        if self.gray_relational_degrees is None:
            self.calculate_gray_correlation()

        # 按关联度排序
        sorted_degrees = self.gray_relational_degrees.sort_values(ascending=False)

        # 排除参考序列自身
        if self.reference_column in sorted_degrees.index:
            sorted_degrees = sorted_degrees.drop(self.reference_column)

        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_degrees.index, sorted_degrees.values, color='skyblue')
        plt.title('各因子与旅游收入的灰色关联度', fontproperties=font, fontsize=15)
        plt.xlabel('因子', fontproperties=font, fontsize=12)
        plt.ylabel('灰色关联度', fontproperties=font, fontsize=12)
        plt.ylim(0, 1)

        # 在柱子上标注关联度值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontproperties=font)

        plt.xticks(rotation=45, ha='right', fontproperties=font)
        plt.tight_layout()

        return plt

    def analyze_results(self):
        """
        分析关联度结果并生成建议
        """
        if self.gray_relational_degrees is None:
            self.calculate_gray_correlation()

        # 按关联度排序
        sorted_degrees = self.gray_relational_degrees.sort_values(ascending=False)

        # 排除参考序列自身
        if self.reference_column in sorted_degrees.index:
            sorted_degrees = sorted_degrees.drop(self.reference_column)

        # 以表格形式打印灰色关联度分析结果
        result_df = pd.DataFrame({"因子": sorted_degrees.index, "关联度": sorted_degrees.values})
        print("灰色关联度分析结果：")
        print(result_df)

        print("\n重要性排序：")
        importance_ranking = "\n".join([f"{i}. {factor}" for i, (factor, _) in enumerate(sorted_degrees.items(), 1)])
        print(importance_ranking)

        # 生成建议
        print("\n旅游发展建议：")
        print(f"1. 重点关注对旅游收入影响最大的因子：{sorted_degrees.index[0]}")
        print(f"2. 针对高关联度因子({', '.join(sorted_degrees.index[:2])})制定发展策略")
        print(f"3. 考虑关联度相对较低的因子({sorted_degrees.index[-1]})的改进空间")

        return sorted_degrees


# 示例数据 - 请替换为实际数据
def generate_sample_data():
    """生成示例数据用于演示"""
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

    # 旅游收入（参考序列）
    tourism_revenue = [5514.90
                       , 6178.00
                       , 6927.38
                       , 4379.49
                       , 6028.28
                       , 5552.81
                       , 7584.80
                       ]

    # 交通因子
    railway_mileage = [4216
                       , 4341
                       , 5165
                       , 5185
                       , 5227
                       , 5603
                       , 5679.5
                       ]  # 铁路里程(公里)
    highway_mileage = [269484
                       , 275039
                       , 289029
                       , 289960
                       , 296922
                       , 302178
                       , 307566
                       ]  # 公路里程(公里)

    # 游客人数因子
    tourist_number = [63499.86
                      , 53556.28
                      , 60143.70
                      , 43694.43
                      , 60621.31
                      , 58460.72
                      , 77160.24
                      ]  # 游客人数(万人次)

    # 可支配收入因子
    disposable_income = [31889
                         , 34455
                         , 37601
                         , 36706
                         , 40278
                         , 42626
                         , 44990
                         ]  # 人均可支配收入(元)

    # 接待因子（总和数据）
    reception_factor = [1130
                        , 1403
                        , 1666
                        , 1898
                        , 2298
                        , 2586
                        , 3230
                        ]  # 酒店和饭店总数(家)

    # 创建数据框
    data = pd.DataFrame({
        '年份': years,
        '旅游收入': tourism_revenue,
        '铁路里程': railway_mileage,
        '公路里程': highway_mileage,
        '游客人数': tourist_number,
        '人均可支配收入': disposable_income,
        '接待因子': reception_factor
    })

    data.set_index('年份', inplace=True)
    return data


# 主函数
def main():
    # 生成示例数据 - 实际使用时请替换为真实数据
    data = generate_sample_data()

    # 创建灰色关联模型实例
    model = GrayCorrelationModel(data, reference_column='旅游收入')

    # 数据预处理
    processed_data = model.preprocess_data(method='initial')
    print("\n预处理后的数据:")
    print(processed_data.round(4))

    # 计算灰色关联度
    gray_relational_degrees = model.calculate_gray_correlation(rho=0.5)
    print("\n灰色关联度:")
    print(gray_relational_degrees.round(4))

    # 分析结果
    sorted_degrees = model.analyze_results()

    # 可视化结果
    plt = model.plot_correlation_results()
    plt.show()


if __name__ == "__main__":
    main()