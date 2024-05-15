import numpy as np

# Confusion matrix
'''
confusion_matrix = np.array([
    [ 681 ,  43  , 56 ,  19  ,  9  ,  6  , 58 ,  11 ,  42 ,  18  ,  3],
 [  78 , 936 , 102 ,  31 ,  52 ,  38 ,  45 ,  72 ,   4 ,  39  ,  1],
 [  80 ,  66 , 658 , 112 , 214 ,  13  , 41 ,  74  , 41  ,  8  ,  0],
 [  72 ,  27 , 130 , 533 ,  37  ,  8  ,  8  , 11  , 25  ,  2  ,  0],
 [  13 ,  44 , 292  , 71 , 306  ,  7  , 15  ,  6  ,  7  ,  3  ,  0],
 [  16 ,  55 ,  64  ,  7 ,  40  , 86  , 56  , 28  , 11  , 16  ,  1],
 [  26 , 100  , 33  ,  6 ,  15  , 13 ,1274  , 18  , 10  , 13  ,  5],
 [  50 ,  40 ,  78  , 56 ,  14  , 15 ,  26  ,360  , 10  ,  3  ,  0],
 [ 137 ,  17 ,  22  ,  6 ,   2  ,  3 ,   3  , 14  ,351  , 14  ,  0],
 [  21 ,  67 ,  32  ,  5 ,  14  ,  7 ,  11  , 18  ,  3  ,572  ,  0],
 [   6 ,   9 ,  22  ,  9 ,   4  ,  2 ,  19  ,  4  ,  3  ,  0  ,  2]
])
'''
confusion_matrix = np.array([

 [ 936 , 102 ,  31 ,  52 ,  38 ,  45 ,  72 ,   4 ,  39  ,  1],
 [  66 , 658 , 112 , 214 ,  13  , 41 ,  74  , 41  ,  8  ,  0],
 [  27 , 130 , 533 ,  37  ,  8  ,  8  , 11  , 25  ,  2  ,  0],
 [  44 , 292  , 71 , 306  ,  7  , 15  ,  6  ,  7  ,  3  ,  0],
 [  55 ,  64  ,  7 ,  40  , 86  , 56  , 28  , 11  , 16  ,  1],
 [  100  , 33  ,  6 ,  15  , 13 ,1274  , 18  , 10  , 13  ,  5],
 [  40 ,  78  , 56 ,  14  , 15 ,  26  ,360  , 10  ,  3  ,  0],
 [ 17 ,  22  ,  6 ,   2  ,  3 ,   3  , 14  ,351  , 14  ,  0],
 [  67 ,  32  ,  5 ,  14  ,  7 ,  11  , 18  ,  3  ,572  ,  0],
 [   9 ,  22  ,  9 ,   4  ,  2 ,  19  ,  4  ,  3  ,  0  ,  2]
])

# Calculate precision, recall, and F1 score for each class
precisions = []
recalls = []
f1_scores = []

for i in range(len(confusion_matrix)):
    tp = confusion_matrix[i, i]
    fp = np.sum(confusion_matrix[:, i]) - tp
    fn = np.sum(confusion_matrix[i, :]) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Compute weighted average of F1 scores based on support
supports = np.sum(confusion_matrix, axis=1)
weighted_f1 = np.average(f1_scores, weights=supports)

print("Weighted F1 Score:", weighted_f1)
