# GeneMLP Five-Fold Cross-Validation 



A transparent, leakage-aware GeneMLP reference implementation for binary
classification with complete fold-level confusion counts, macro-precision
calculations, and mean ± sample-standard-deviation reporting.

Scope: This repository is a standalone methodological and statistical
audit example. It uses synthetic data and fictional class labels. It does not
contain participant data, institutional information, private experiment
records, trained weights, or manuscript results.

Purpose

GeneMLP_CV_Fold_Audit.py makes every step behind a reported five-fold
cross-validation value auditable. For each validation fold, it reports:

the number of observations from each class;

the complete $2\times2$ confusion matrix;

TP, FP, FN, and TN for each class under a one-vs-rest interpretation;

class-specific precision;

macro precision;

macro recall;

macro F1;

ordinary accuracy; and

the unrounded values used to calculate the five-fold mean and sample SD.

The script also independently cross-checks:

the manual macro-precision calculation against scikit-learn; and

the manually calculated sample SD against NumPy with ddof=1.

Repository Contents

File

Description

GeneMLP_CV_Fold_Audit.py

Model, cross-validation, fold-level audit calculations, console report, and optional CSV export

README.md

Methodological, statistical, and execution documentation

cv_fold_audit_example.csv

Generated after running the synthetic demonstration

Workflow

flowchart TD
    A[Development data only] --> B[Stratified five-fold split]
    B --> C[Fit scaler on training fold]
    C --> D[Transform validation fold]
    D --> E[Initialize a fresh GeneMLP]
    E --> F[Train on training fold only]
    F --> G[Predict validation fold]
    G --> H[Confusion counts and metrics]
    H --> I[Mean and sample SD]

GeneMLP Architecture

For an input vector containing $p$ features and two outcome classes, the model
is:

$$
\mathbf{x}\in\mathbb{R}^{p}
\rightarrow \operatorname{Linear}(p,512)
\rightarrow \operatorname{ReLU}
\rightarrow \operatorname{Dropout}(0.30)
$$

$$
\rightarrow \operatorname{Linear}(512,256)
\rightarrow \operatorname{ReLU}
\rightarrow \operatorname{Dropout}(0.30)
\rightarrow \operatorname{Linear}(256,2).
$$

The output layer returns logits. Training uses cross-entropy loss and Adam.
A new model, loss object, and optimizer are created independently inside every
fold.

Default settings:

Parameter

Value

Cross-validation folds

5

Random seed

42

First hidden layer

512

Second hidden layer

256

Dropout

0.30

Learning rate

$10^{-3}$

Weight decay

$10^{-4}$

Batch size

8

Training epochs

100

The synthetic demonstration uses five epochs only so that it runs quickly.
An actual audit must use the exact prespecified settings associated with the
reported result.

Five-Fold Cross-Validation

The splitter is:

StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

Stratification approximately preserves the class distribution across the five
validation folds. Fold sizes can differ by one when the development sample size
is not divisible by five.

The included fictional example contains 41 synthetic observations with class
totals of 16 and 25. Consequently, its validation-fold composition is:

one fold with 9 observations: 4 from one class and 5 from the other; and

four folds with 8 observations: 3 from one class and 5 from the other.

These are synthetic demonstration counts and are not manuscript results.

Leakage Controls

Potential source of leakage

Control in the script

Scaling before splitting

Fold indices are created before standardization

Validation data influencing scaling

StandardScaler.fit_transform is used only on the training fold

Reusing fitted scaling parameters incorrectly

Validation data receive scaler.transform only

Reusing model weights

A new GeneMLP is initialized inside every fold

Reusing optimizer state

A new Adam optimizer is created inside every fold

Training during validation

Validation uses model.eval() and torch.no_grad()

Overlapping partitions

Training and validation indices are explicitly checked for overlap

Premature rounding

Fold metrics remain unrounded until final display

Incorrect SD denominator

Sample SD is calculated with ddof=1

These safeguards prevent direct train-validation leakage inside the presented
function. They do not validate preprocessing performed before the function is
called.

Any learned operation—including imputation, normalization, batch correction,
feature selection, differential-expression filtering, pathway selection, PCA,
or representation learning—must also be fitted using training data only.

Confusion-Matrix Convention

The script uses rows for actual classes and columns for predicted classes:

$$
\begin{array}{c|cc}
& \text{Predicted Class 0} & \text{Predicted Class 1} \
\hline
\text{Actual Class 0} & a & b \
\text{Actual Class 1} & c & d
\end{array}
$$

Equivalently:

$$
\mathbf{C}=
\begin{bmatrix}
a & b\
c & d
\end{bmatrix}.
$$

Because macro metrics treat each class as the positive class in turn, the raw
counts are interpreted as follows:

Quantity

Class 0 as positive

Class 1 as positive

TP

$a$

$d$

FP

$c$

$b$

FN

$b$

$c$

TN

$d$

$a$

This class-specific interpretation avoids ambiguity from reporting only one set
of TP, FP, FN, and TN values for a binary macro metric.

Precision Calculations

Class 0 precision

The total number predicted as Class 0 is $a+c$. Therefore:

$$
P_0=\frac{a}{a+c}.
$$

Class 1 precision

The total number predicted as Class 1 is $b+d$. Therefore:

$$
P_1=\frac{d}{b+d}.
$$

Macro precision

Macro precision gives both classes equal weight:

$$
P_{\mathrm{macro}}=\frac{P_0+P_1}{2}
=\frac{1}{2}\left(\frac{a}{a+c}+\frac{d}{b+d}\right).
$$

The implementation uses:

precision_score(
    y_true,
    y_pred,
    labels=[0, 1],
    average="macro",
    zero_division=0,
)

If a class receives no predictions, its precision denominator is zero.
zero_division=0 assigns that class a precision of zero instead of returning an
undefined value.

Recall Calculations

Recall uses the number of actual observations from each class as its
denominator.

For Class 0:

$$
R_0=\frac{a}{a+b}.
$$

For Class 1:

$$
R_1=\frac{d}{c+d}.
$$

Macro recall is:

$$
R_{\mathrm{macro}}=\frac{R_0+R_1}{2}.
$$

F1 Calculations

For class $j\in{0,1}$:

$$
F1_j=\frac{2P_jR_j}{P_j+R_j}.
$$

Macro F1 is the arithmetic mean of the two class-specific F1 scores:

$$
F1_{\mathrm{macro}}=\frac{F1_0+F1_1}{2}.
$$

Macro F1 is not generally equal to the F1 score calculated from macro precision
and macro recall.

Accuracy

Accuracy is the fraction of all correctly classified observations:

$$
\operatorname{Accuracy}=\frac{a+d}{a+b+c+d}.
$$

Accuracy is not a macro-averaged metric. A reported macro-precision value should
not be described as accuracy.

Why Macro Precision Is Reported

Macro precision calculates precision independently for both classes and then
weights the classes equally. This prevents the larger class from dominating the
reported precision.

In standard single-label classification, micro precision, micro recall, and
micro F1 reduce to the same value as accuracy. Macro metrics therefore provide
more explicit information about class-balanced performance.

Five-Fold Mean

Let $P_k$ be the unrounded macro precision calculated for validation fold $k$.
For $K=5$ folds:

$$
\bar{P}=\frac{1}{K}\sum_{k=1}^{K}P_k
=\frac{P_1+P_2+P_3+P_4+P_5}{5}.
$$

The script calculates an unweighted arithmetic mean of the five fold-level
macro-precision values. This is not necessarily identical to calculating macro
precision once after pooling all out-of-fold predictions.

Sample Standard Deviation

The reported fold-to-fold SD is the sample standard deviation:

$$
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(P_k-\bar{P})^2}.
$$

For five folds:

$$
s=\sqrt{\frac{
(P_1-\bar{P})^2+
(P_2-\bar{P})^2+
(P_3-\bar{P})^2+
(P_4-\bar{P})^2+
(P_5-\bar{P})^2
}{4}}.
$$

The NumPy implementation is:

sample_sd = values.std(ddof=1)

ddof=1 uses the $K-1$ denominator required for sample SD. NumPy's default
ddof=0 instead calculates population SD.

For five values, the relationship is:

$$
s_{\mathrm{sample}}
=s_{\mathrm{population}}\sqrt{\frac{5}{4}}
\approx1.118,s_{\mathrm{population}}.
$$

Thus, sample SD is approximately 11.8% larger than population SD when $K=5$.

Worked SD Example

Consider five fold values:

$$
[0.80,;0.80,;0.80,;0.80,;1.00].
$$

The mean is:

$$
\bar{P}=\frac{0.80+0.80+0.80+0.80+1.00}{5}=0.84.
$$

The sum of squared deviations is:

$$
4(0.80-0.84)^2+(1.00-0.84)^2=0.032.
$$

The value $0.032$ is not the SD. It is the sum of squared deviations. The
sample SD is:

$$
s=\sqrt{\frac{0.032}{5-1}}
=\sqrt{0.008}
=0.08944.
$$

The correctly reported result is therefore approximately:

$$
0.84\pm0.09.
$$

Interpretation of a Small SD

A small cross-validation SD indicates that the observed fold-level values were
similar. It can support the statement that performance was consistent across
the five development folds.

A small SD does not independently prove:

robustness to a new population;

external validity;

calibration;

individual-prediction certainty; or

freedom from upstream leakage.

Recommended wording after the raw calculations have been verified:

The relatively small cross-validation standard deviation indicates
consistent performance across the five development folds.

Requirements

Python 3.10 or later

NumPy

scikit-learn

PyTorch

Install the dependencies with:

python -m pip install numpy scikit-learn torch

Run the Synthetic Audit Example

python GeneMLP_CV_Fold_Audit.py

The script will:

generate an anonymized synthetic binary dataset;

run stratified five-fold cross-validation;

print fold class totals and confusion matrices;

print class-specific precision equations;

print macro precision and accuracy for every fold;

print the five unrounded macro-precision values;

show the exact mean and sample-SD calculations; and

create cv_fold_audit_example.csv.

Use Development Arrays

import numpy as np

from GeneMLP_CV_Fold_Audit import Config, run_cross_validation

# These filenames are fictional examples. Supply development data only and
# keep the external test set separate.
X_development = np.load(
    "example_feature_matrix.npy",
    allow_pickle=False,
).astype(np.float32)

y_development = np.load(
    "example_class_labels.npy",
    allow_pickle=False,
)

config = Config(
    n_splits=5,
    seed=42,
    hidden1=512,
    hidden2=256,
    dropout=0.30,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=8,
    epochs=100,
)

summary, fold_records = run_cross_validation(
    X=X_development,
    y=y_development,
    config=config,
    audit_csv="cv_fold_audit.csv",
    verbose=True,
)

To reproduce a reported result, the following must match the original analysis:

development observations;

outcome labels;

feature representation;

upstream preprocessing;

random seed and fold assignments;

model architecture;

optimizer settings;

number of epochs; and

every data-dependent selection decision.

Fold-Audit CSV

The optional CSV contains full-precision fold records, including:

fold number and training/validation sizes;

encoded class names;

actual and predicted class totals;

all four confusion-matrix cells;

TP, FP, FN, and TN for each class;

class-specific precision;

macro precision;

macro recall;

macro F1; and

accuracy.

Do not upload a CSV generated from private participant-level data to a public
repository. The included demonstration output is synthetic.

Reproducibility

The script sets random seeds for:

Python;

NumPy;

PyTorch;

CUDA, when available; and

the shuffled training-data loader.

It also requests deterministic PyTorch algorithms and deterministic cuDNN
behavior. Exact bitwise reproducibility may still depend on the operating
system, hardware, CUDA version, PyTorch build, and device-specific numerical
behavior.

No MLOps Components

This repository contains no MLOps integration. It does not include:

Weights & Biases;

hyperparameter sweeps;

automated experiment tracking;

model registries;

cloud services;

deployment workflows; or

automated model selection.

The optional CSV is a local statistical audit record, not an MLOps component.

Statistical Audit Boundary

The code verifies the calculations it performs. It cannot establish that a
previously reported table or figure used the same fold predictions unless the
script is run with the exact original inputs and settings.

The final audit evidence for a reported cell should include:

the five validation-fold class totals;

the five confusion matrices;

both class-specific precision values for every fold;

the five unrounded macro-precision values;

the mean calculation; and

the sample-SD calculation.
