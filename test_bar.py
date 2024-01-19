import matplotlib.pyplot as plt
import seaborn as sns

# 英文类别名
categories = ["Category 1", "Category 2", "Category 3", "Category 4"]
counts = [10, 20, 15, 25]

# 使用Seaborn样式
sns.set(style="whitegrid")

# 创建柱状图
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=categories, y=counts, palette="viridis", ci=None)

# 添加标签和标题
plt.xlabel("Categories", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.title("Beautiful Bar Chart", fontsize=16)

# 在柱状图上显示数值
for index, value in enumerate(counts):
    bar_plot.text(index, value + 0.5, str(value), ha='center', va='bottom')

# 调整柱状图的比例，可以根据需要进行调整
bar_plot.set_ylim(0, max(counts) + 5)

# 保存图像
plt.savefig('bar_chart_beautiful.png', bbox_inches='tight', dpi=300)
plt.show()
