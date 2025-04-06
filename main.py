
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from ann_model import train_and_save_model
from fuzzy_logic import fuzzy_predict
from utils import load_and_preprocess
from ann_model import train_and_save_model, predict_depression
import csv
import re
import matplotlib.pyplot as plt
class DepressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự đoán nguy cơ trầm cảm")
        self.root.geometry("600x400")
        self.build_gui()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill='both', expand=True)

        self.btn_load = ttk.Button(frame, text="Tải và xử lý dữ liệu", command=self.load_data)
        self.btn_load.pack(pady=10)

        self.btn_train = ttk.Button(frame, text="Huấn luyện mô hình ANN", command=self.train_model, state='disabled')
        self.btn_train.pack(pady=10)

        self.btn_fuzzy = ttk.Button(frame, text="Dự đoán bằng Logic mờ", command=self.predict_fuzzy, state='disabled')
        self.btn_fuzzy.pack(pady=10)

        self.btn_chart = ttk.Button(frame, text="Hiển thị Biểu đồ", command=self.show_chart, state='disabled')
        self.btn_chart.pack(pady=10)
# tao sửa vẫn để lại code cũ cho xem nhé , phần nhiều là mày xử lý data sai thôi chả có cc gì cả
    def load_data(self):
        try:
            # df = pd.read_csv("Student-Depression-Dataset.csv")
            # self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = load_and_preprocess(df)
            self.X_train, self.X_test, self.y_train, self.y_test, self.scaler, self.encoders = load_and_preprocess("Student-Depression-Dataset.csv")
            messagebox.showinfo("Thành công", "Tải và xử lý dữ liệu thành công!")
            self.btn_train.config(state='normal')
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def train_model(self):
        self.model = train_and_save_model(self.X_train, self.y_train)
        self.y_pred = predict_depression(self.model, self.X_test)
        # self.model, self.y_pred = train_and_save_model(self.X_train, self.y_train, self.X_test, self.y_test)
        messagebox.showinfo("Hoàn tất", "Huấn luyện ANN hoàn tất! Kiểm tra kết quả trên terminal.")
        self.btn_fuzzy.config(state='normal')

    def predict_fuzzy(self):
        result = fuzzy_predict(self.X_test)
        pd.DataFrame({"Fuzzy Result": result}).to_csv("fuzzy_results.csv", index=False)
        messagebox.showinfo("Xong!", "Dự đoán logic mờ hoàn tất, lưu tại fuzzy_results.csv")
        self.btn_chart.config(state='normal')
    def parse_value(self, s):
        try:
            if 'np.float64' in s:
                return float(re.search(r"\((.*?)\)", s).group(1))
            return float(s)
        except:
            return None

    def read_and_parse_data(self, file_path):
        all_values = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].strip().startswith('['):
                    continue
                clean_row = row[0].strip('[]').split(', ')
                parsed_row = [self.parse_value(x) for x in clean_row]
                parsed_row = [x for x in parsed_row if x is not None]
                all_values.extend(parsed_row)
        return all_values

    def calculate_distribution(self, all_values):
        thresholds = {
            "Không Trầm cảm": (0.0, 0.4),
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
        return percentages

    def plot_chart(self, percentages):
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
            messagebox.showerror("Lỗi", "Không có dữ liệu hợp lệ để vẽ biểu đồ.")

    def show_chart(self):
        file_path = 'fuzzy_results.csv'  # Đổi lại thành đường dẫn đến file của bạn
        all_values = self.read_and_parse_data(file_path)
        percentages = self.calculate_distribution(all_values)
        self.plot_chart(percentages)

if __name__ == '__main__':
    root = tk.Tk()
    app = DepressionApp(root)
    root.mainloop()
