import numpy as np

# String representation of the confusion matrix
confusion_matrix_str = "[[65  5 17 15  2  2 48  4 14 22  0] [ 8 70 37  9 11  1 35 11  5 12  0] [14 21 87 22  9  3 14 23  6  3  0] [22  7 59 57 11  4 22  6  1  3  0] [ 8 26 73 19 19  2  9  8  2  2  0] [ 3  4 12  3  1 10 13  7  3  2  0] [16 10 19 17  4  2 72  6  6  6  0] [ 8  7 26  5  8  6 19 44  1  2  0] [25  2  7  5  1  0  5  4 29  0  3] [ 1 16 18  2  6  1 26  1  0 39  0] [ 0  0  2  1  0  1  1  1  0  1  1]]"
# Parse the string into a numpy array
confusion_matrix = np.array(eval(confusion_matrix_str))

# Calculate true positives, false positives, and false negatives for class 0
TP = confusion_matrix[0, 0]
FP = np.sum(confusion_matrix[1:, 0])
FN = np.sum(confusion_matrix[0, 1:])

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
