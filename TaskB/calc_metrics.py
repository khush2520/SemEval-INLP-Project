import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

df = pd.read_csv('predictions1.csv')

y_true = df['correct_label']
y_pred = df['predicted_label']

cm = confusion_matrix(y_true, y_pred)

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# \item Accuracy: 0.7940789473684211
# \item F1 Score: 0.7930376743193894
# \item Precision: 0.7928708882822995
# \item Recall: 0.7940298968837314

# \item Accuracy: 0.8102631578947368
# \item F1 Score: 0.809402006811925
# \item Precision: 0.809365845199265
# \item Recall: 0.8102631578947368