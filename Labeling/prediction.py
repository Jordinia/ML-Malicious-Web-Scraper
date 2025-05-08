import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("netpro_raw_7k_val.csv", encoding="utf-8")

predictions = pd.read_csv("NOSFT_netpro_raw_7k_val_label.csv", encoding="utf-8")

# Step 5: Evaluation
true_labels = df['Label']

predictions = predictions['Label']

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
report = classification_report(true_labels, predictions)

# Display on terminal
print("Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nDetailed classification report:\n")
print(report)

# Save metrics to .txt
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Evaluation Metrics:\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")
    f.write("Detailed classification report:\n")
    f.write(report)

# Step 6: Add predictions to DataFrame and export
df['predicted_label'] = predictions
df.to_csv("classified_output_val.csv", index=False)
print("\nClassification complete. Output saved to 'classified_output.csv' and 'classification_report.txt'")
