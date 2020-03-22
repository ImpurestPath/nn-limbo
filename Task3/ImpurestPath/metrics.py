def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp,fp,fn,tn = 0,0,0,0
    for index, elem in enumerate(prediction):
        if elem:
            if ground_truth[index]:
                tp += 1
            else:
                fp += 1
        else:
            if ground_truth[index]:
                fn += 1
            else:
                tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    truth = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            truth += 1

    return truth / prediction.shape[0]
