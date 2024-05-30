import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 设置matplotlib绘图时的字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def load_data(file_path):
    try:
        train_df = pd.read_csv(file_path)
        return train_df
    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def extract_features(df):
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # 根据票价进行分段处理
    bins = [0, 7, 12, 30, 870]
    labels = [0, 1, 2, 3]  # 使用0, 1, 2, 3表示不同的票价段
    df['FareBin'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)

    return df


def split_and_save_by_gender(df, male_file='train_male.csv', female_file='train_female.csv'):
    try:
        df = extract_features(df)

        male_df = df[df['Sex'] == 'male']
        female_df = df[df['Sex'] == 'female']

        male_df = male_df.sort_values(by='Fare', ascending=False)
        female_df = female_df.sort_values(by='Fare', ascending=False)

        if os.path.exists(male_file):
            os.remove(male_file)
        if os.path.exists(female_file):
            os.remove(female_file)

        male_df.to_csv(male_file, index=False)
        female_df.to_csv(female_file, index=False)
    except Exception as e:
        print(f"处理或保存数据时出错: {e}")


train_df = load_data("./train.csv")
if train_df is not None:
    # 填补缺失值
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    train_df = train_df.drop(columns=['Cabin'])

    split_and_save_by_gender(train_df)

# 显示处理后的数据的前几行
if train_df is not None:
    print(train_df[['Name', 'Surname', 'Title', 'Fare', 'FareBin']].head())

# 统计每个港口的存活人数、票价分布、存活率
embarked_survived = train_df.groupby(['Embarked', 'Survived']).size().unstack().fillna(0)
embarked_fare_stats = train_df.groupby(['Embarked', 'FareBin'], observed=False).size().unstack().fillna(0)
total_counts = train_df['Embarked'].value_counts()

# 计算存活率
embarked_survived['Survival Rate'] = embarked_survived[1] / (embarked_survived[0] + embarked_survived[1])

# 合并数据
merged_data = embarked_survived.copy()
merged_data = merged_data.rename(columns={0: 'Not Survived', 1: 'Survived'})
merged_data = merged_data.join(embarked_fare_stats, lsuffix='_count', rsuffix='_fare')

# 重命名列以便更好地理解
merged_data = merged_data.rename(columns={
    '0-7': 'Fare 0-7',
    '7-12': 'Fare 7-12',
    '12-30': 'Fare 12-30',
    '30-870': 'Fare 30-870'
})

# 保存整合后的数据到CSV文件
merged_data.to_csv('embarked_summary.csv')

print("\n港口存活人数：")
print(embarked_survived)

print("\n不同票价区间的乘客人数：")
print(embarked_fare_stats)

print("\n港口总人数：")
print(total_counts)

# 票价分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Fare'], bins=50, kde=True)
plt.title('票价分布')
plt.xlabel('票价')
plt.ylabel('频率')
plt.show()

# 港口分布饼图
embarked_counts = train_df['Embarked'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(embarked_counts, labels=embarked_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('港口分布')
plt.show()

# 基于港口的购票数量的票价分布
plt.figure(figsize=(12, 8))
sns.countplot(x='Embarked', hue='FareBin', data=train_df)
plt.title('港口的购票数量')
plt.xlabel('港口')
plt.ylabel('购票数量')
plt.legend(title='FareBin', labels=['0-7', '7-12', '12-30', '30-870'])
plt.show()

# 港口存活人数分布
plt.figure(figsize=(10, 6))
embarked_survived[[0, 1]].plot(kind='bar', stacked=True)
plt.title('港口存活人数')
plt.xlabel('港口')
plt.ylabel('存活人数')
plt.legend(['未存活', '存活'])
plt.show()

# 港口存活率
plt.figure(figsize=(10, 6))
embarked_survived['Survival Rate'].plot(kind='bar')
plt.title('港口存活率')
plt.xlabel('港口')
plt.ylabel('存活率')
plt.ylim(0, 1)
plt.show()

