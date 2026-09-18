
# GeneMLP Five-Fold CV 

A transparent GeneMLP reference implementation for leakage-aware stratified
five-fold cross-validation, fold-level confusion counts, macro-precision
reporting, and mean ± sample-standard-deviation verification.

Scope: This is a methodological audit example using synthetic data and
fictional labels. It contains no participant data, private paths, trained
weights, institutional information, or manuscript results.

What the Script Reports

GeneMLP_CV_Fold_Review.py reports, for each validation fold:

class totals;

the complete $2\times2$ confusion matrix;

TP, FP, FN, and TN for both classes;

class-specific precision;

macro precision, macro recall, and macro F1;

ordinary accuracy; and

the unrounded values used for the five-fold mean and sample SD.

The script cross-checks the manual calculations against scikit-learn and NumPy.
