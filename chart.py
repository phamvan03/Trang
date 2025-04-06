import csv
import re
import matplotlib.pyplot as plt

def parse_value(s):
    try:
        if 'np.float64' in s:
            return float(re.search(r"\((.*?)\)", s).group(1))
        return float(s)
    except:
        return None 
all_values = []
with open('fuzzy_results.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or not row[0].strip().startswith('['):
            continue
        clean_row = row[0].strip('[]').split(', ')
        parsed_row = [parse_value(x) for x in clean_row]
        parsed_row = [x for x in parsed_row if x is not None]
        all_values.extend(parsed_row)
thresholds = {
    "Không Trâm cảm": (0.0, 0.4),
    "Trầm cảm": (0.6, 1.0)
}
distribution = {category: 0 for category in thresholds}
for val in all_values:
    for category, (low, high) in thresholds.items():
        if low <= val < high:
            distribution[category] += 1
            break
total = len(all_values)
percentages = {k: (v / total) * 100 for k, v in distribution.items()} if total != 0 else {}
if percentages:
    categories = list(percentages.keys())
    values = list(percentages.values())
    colors = ['#98FB98', '#FFA07A', '#FF6347']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=colors, edgecolor='black')

    plt.title('PHÂN PHỐI GIÁ TRỊ THEO TRẠNG THÁI', fontweight='bold', pad=20)
    plt.xlabel('Nhóm', fontsize=12)
    plt.ylabel('Tỷ lệ (%)', fontsize=12)
    plt.ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
else:
    print("Không có dữ liệu hợp lệ để vẽ biểu đồ!")