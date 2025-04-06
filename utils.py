import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path):

    df = pd.read_csv(csv_path)

  
    print("Các tên cột trong DataFrame:", df.columns)


    if 'Profession' in df.columns:
        print("Kiểu của cột 'Profession':", df['Profession'].dtype) 
        print("Giá trị duy nhất trong cột 'Profession':", df['Profession'].unique())  
    else:
        raise ValueError("Cột 'Profession' không có trong dữ liệu!")


    if not isinstance(df['Profession'], pd.Series):
        raise TypeError("'Profession' không phải là một pandas Series!")


    df = df[df['Profession'] == 'Student']
    df = df.dropna()

    columns_to_drop = ['City', 'Work Pressure', 'Job Satisfaction']
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    if missing_columns:
        print(f"Cảnh báo: Các cột sau không có trong dữ liệu: {missing_columns}")
    df = df.drop(columns=columns_to_drop, errors='ignore')  


    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = df[col].apply(lambda x: le.fit_transform([x])[0])  
        label_encoders[col] = le

    X = df.drop(columns=['Depression'])  
    y = df['Depression']                 

    if X.isnull().any().any():
        raise ValueError("Dữ liệu chứa giá trị NaN trong features!")

    if y.isnull().any():
        raise ValueError("Dữ liệu chứa giá trị NaN trong label 'Depression'!")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders
