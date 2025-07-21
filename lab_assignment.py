import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load and split purchase data
def load_purchase_data(file_path):
    data_frame = pd.read_excel(file_path, sheet_name="Purchase data")
    item_matrix = data_frame.iloc[:, 1:4].values   # Features
    cost_vector = data_frame.iloc[:, 4].values.reshape(-1, 1)  # Labels
    return item_matrix, cost_vector

# Analyze matrix properties and compute cost estimates
def analyze_purchase_data(item_matrix, cost_vector):
    dimension = item_matrix.shape[1]
    row_count = item_matrix.shape[0]
    matrix_rank = np.linalg.matrix_rank(item_matrix)
    estimated_costs = np.linalg.pinv(item_matrix).dot(cost_vector)
    return dimension, row_count, matrix_rank, estimated_costs

# Classify based on actual cost
def label_customers(cost_vector, limit=200):
    return ["RICH" if val > limit else "POOR" for val in cost_vector.flatten()]

# Heuristic classifier using total item sum
def predict_customers(item_matrix):
    total_items = item_matrix.sum(axis=1)
    average_amount = total_items.mean()
    predictions = ["RICH" if total > average_amount else "POOR" for total in total_items]
    return predictions

# Load IRCTC stock price data
def fetch_stock_data(file_path):
    stock_data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    return stock_data

# Calculate price statistics
def price_statistics(stock_data):
    price_list = stock_data["Price"].dropna().tolist()
    mean_val = statistics.mean(price_list)
    variance_val = statistics.variance(price_list)
    return mean_val, variance_val

# Perform conditional stats (day, month-based)
def stock_condition_analysis(stock_data):
    price_change = stock_data["Chg%"].dropna()
    stock_wed = stock_data[stock_data["Day"] == "Wed"]
    stock_apr = stock_data[stock_data["Month"] == "Apr"]
    
    prob_loss = len(price_change[price_change < 0]) / len(price_change)
    prob_profit = len(stock_data[stock_data["Chg%"] > 0]) / len(stock_data)
    
    prob_profit_wed = (
        len(stock_wed[stock_wed["Chg%"] > 0]) / len(stock_wed)
        if len(stock_wed) > 0 else 0
    )
    
    return stock_wed["Price"].mean(), stock_apr["Price"].mean(), prob_loss, prob_profit, prob_profit_wed

# Scatterplot of Chg% vs. Day
def plot_stock_change(stock_data):
    sns.scatterplot(data=stock_data, x="Day", y="Chg%")
    plt.title("Price Change Percentage by Day")
    plt.tight_layout()
    plt.show()

# Load thyroid dataset
def load_thyroid(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Determine types of data
def detect_column_types(data_frame):
    return {
        column_name: 'categorical' if data_frame[column_name].dtype == object else 'numeric'
        for column_name in data_frame.columns
    }

# Encode all text columns into numerics
def label_encode_all(data_frame):
    df_encoded = data_frame.copy()
    encoder = LabelEncoder()
    for column_name in df_encoded.columns:
        if df_encoded[column_name].dtype == object:
            df_encoded[column_name] = encoder.fit_transform(df_encoded[column_name].astype(str))
    return df_encoded

# Check for nulls and describe statistics
def get_info_summary(data_frame):
    return data_frame.isnull().sum(), data_frame.describe()

# Binary similarity coefficients
def binary_similarity(row1, row2):
    f11 = sum(a == b == 1 for a, b in zip(row1, row2))
    f00 = sum(a == b == 0 for a, b in zip(row1, row2))
    f10 = sum(a == 1 and b == 0 for a, b in zip(row1, row2))
    f01 = sum(a == 0 and b == 1 for a, b in zip(row1, row2))
    
    jaccard_score = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
    smc_score = (f11 + f00) / (f11 + f00 + f10 + f01)
    
    return jaccard_score, smc_score

# Vector cosine similarity
def cosine_match(row1, row2):
    arr1, arr2 = np.array(row1, dtype=float), np.array(row2, dtype=float)
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

# Heatmaps of similarity between first n rows
def plot_similarity_heatmaps(data_frame, n=20):
    sample_data = label_encode_all(data_frame.head(n).fillna(0))

    jaccard_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec1 = sample_data.iloc[i].values
            vec2 = sample_data.iloc[j].values
            jaccard_matrix[i][j], smc_matrix[i][j] = binary_similarity(vec1, vec2)
            cosine_matrix[i][j] = cosine_match(vec1, vec2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(jaccard_matrix, ax=axes[0]); axes[0].set_title("Jaccard")
    sns.heatmap(smc_matrix, ax=axes[1]); axes[1].set_title("SMC")
    sns.heatmap(cosine_matrix, ax=axes[2]); axes[2].set_title("Cosine Similarity")
    plt.tight_layout()
    plt.show()

# Impute missing values column-wise
def fill_missing_values(data_frame):
    for column_name in data_frame.columns:
        if data_frame[column_name].dtype != object:
            skewness = data_frame[column_name].skew()
            fill_val = data_frame[column_name].median() if abs(skewness) > 1 else data_frame[column_name].mean()
            data_frame[column_name] = data_frame[column_name].fillna(fill_val)
        else:
            data_frame[column_name] = data_frame[column_name].fillna(data_frame[column_name].mode()[0])
    return data_frame

# Min-max normalization for numeric columns
def min_max_normalize(data_frame):
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns
    data_frame[numeric_columns] = MinMaxScaler().fit_transform(data_frame[numeric_columns])
    return data_frame

# Boxplots of numeric features
def draw_boxplots(data_frame):
    num_columns = data_frame.select_dtypes(include="number").columns
    for column_name in num_columns:
        plt.figure(figsize=(4, 6))
        sns.boxplot(y=data_frame[column_name].dropna())
        plt.title(f"Boxplot: {column_name}")
        plt.tight_layout()
        plt.show()

# --- Main ---
def main():
    file_path = r"C:\Users\Thirshith\Downloads\Lab-Session-Data.xlsx"

    item_matrix, cost_vector = load_purchase_data(file_path)
    dimension, count, matrix_rank, estimated_costs = analyze_purchase_data(item_matrix, cost_vector)

    print("Results - Section A1")
    print("Dimensions:", dimension)
    print("Samples:", count)
    print("Matrix Rank:", matrix_rank)
    print("Estimated Product Costs:", estimated_costs.flatten())

    print("\nSection A2:")
    true_classes = label_customers(cost_vector)
    predicted_classes = predict_customers(item_matrix)
    print("Actual Labels:", true_classes)
    print("Predicted Labels:", predicted_classes)

    print("\nSection A3:")
    stock_data = fetch_stock_data(file_path)
    mean_price, price_var = price_statistics(stock_data)
    wed_mean, apr_mean, loss_prob, profit_prob, wed_profit_prob = stock_condition_analysis(stock_data)
    print(f"Mean Price: {mean_price}, Variance: {price_var}")
    print(f"Wednesday Mean: {wed_mean}, April Mean: {apr_mean}")
    print(f"P(Loss): {loss_prob}, P(Profit): {profit_prob}, P(Profit | Wed): {wed_profit_prob}")
    plot_stock_change(stock_data)

    print("\nSection A4:")
    thyroid_data = load_thyroid(file_path)
    column_types = detect_column_types(thyroid_data)
    null_counts, summary_stats = get_info_summary(thyroid_data)
    print("Types:", column_types)
    print("Null Values:\n", null_counts)
    print("Summary Stats:\n", summary_stats)

    print("\nPlotting Boxplots for Numerical Data:")
    draw_boxplots(thyroid_data)

    print("\nSection A5 & A6:")
    encoded_thyroid = label_encode_all(thyroid_data.fillna(0))
    row1, row2 = encoded_thyroid.iloc[0].values, encoded_thyroid.iloc[1].values
    jaccard_score, smc_score = binary_similarity(row1, row2)
    cos_score = cosine_match(row1, row2)
    print("Jaccard:", jaccard_score)
    print("SMC:", smc_score)
    print("Cosine Similarity:", cos_score)

    print("\nSection A7 - Heatmaps:")
    plot_similarity_heatmaps(thyroid_data)

    print("\nSection A8 - Imputed Values")
    imputed_values = fill_missing_values(thyroid_data.copy())

    print("\nSection A9 - Normalized Data")
    normalized_data = min_max_normalize(imputed_values.copy())

if __name__ == "__main__":
    main()
