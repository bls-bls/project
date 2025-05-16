import pandas as pd
import os
import re

# 读取CSV文件
file_path = 'summerOly_athletes.csv'
df = pd.read_csv(file_path)

# 只保留所需列
columns_to_keep = ['Name', 'Sex', 'NOC', 'Year', 'City', 'Sport', 'Event', 'Medal']
df = df[columns_to_keep]

# 去除包含空值的行
df.dropna(inplace=True)

# 异常值筛除
valid_sex = ['M', 'F']
valid_medals = ['Gold', 'Silver', 'Bronze', '']

# Year 应该在 1896 到 2024
df = df[df['Year'].between(1896, 2024)]

# Sex 仅限 M 或 F
df = df[df['Sex'].isin(valid_sex)]

# Medal 必须是允许的值
df = df[df['Medal'].isin(valid_medals)]

# NOC 必须是三个大写字母
df = df[df['NOC'].apply(lambda x: isinstance(x, str) and bool(re.fullmatch(r'[A-Z]{3}', x)))]

# 创建保存目录
output_dir = 'NOC_CSVs'
os.makedirs(output_dir, exist_ok=True)

# 分组导出
for noc, group in df.groupby('NOC'):
    group_sorted = group.sort_values(by='Year')
    output_file = os.path.join(output_dir, f'{noc}.csv')
    group_sorted.to_csv(output_file, index=False)