import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    sum = 0
    for i in range(len(fpr)-1):
        sum += ((tpr[i] + tpr[i+1])*(fpr[i+1]- fpr[i]))/2
    pass

    return sum