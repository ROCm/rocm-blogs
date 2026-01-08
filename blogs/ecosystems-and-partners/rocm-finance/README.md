---
blogpost: true
blog_title: "Using Gradient Boosting Libraries on MI300X for Financial Risk Prediction"
date: 8 Jan 2026
author: 'Karthik Kashyap Thatipamula, Ish Kool, Yazhini Rajesh, Marco Grond'
thumbnail: 'rocm-finance.png'
tags: 'AI/ML'
category: Applications & models
target_audience: Finance, Developers
key_value_propositions: Financial institutions deal with massive datasets and complex models that demand high computational power. With AMD ROCm and GPU-accelerated libraries like LightGBM and ThunderGBM, we can significantly reduce training time while improving model performance
language: English
myst:
    html_meta:
        "author": "Karthik Kashyap Thatipamula, Ish Kool, Yazhini Rajesh, Marco Grond"
        "description lang=en": "This blog shows how to run LightGBM and ThunderGBM GPU-accelerated training on AMD Instinct MI300X GPUs with ROCm for finance focused workloads."
        "keywords": "finance, gradient boosting, rocm"
        "vertical": "Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Data Science"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Karthik Kashyap Thatipamula, Ish Kool, Yazhini Rajesh, Marco Grond"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Using Gradient Boosting Libraries on MI300X for Financial Risk Prediction

In the world of machine learning, the choice of hardware can significantly impact the performance and efficiency of model training and prediction. Gradient Boosting Machines (GBMs) benefit greatly from GPU parallelization in several key algorithmic steps involving independent, repetitive computations. The most substantial speedup comes from histogram construction and best split searching, as these can be executed in parallel across features and candidate splits using thousands of GPU cores, vastly accelerating tree building. Additionally, the calculation of gradients and Hessians for each data point is naturally parallelizable and well suited to GPU architectures. Other operations—such as leaf value updates, data preprocessing (like quantization and normalization), and batch predictions—can also be distributed efficiently across GPU threads. By exploiting parallelism in these stages, GPUs dramatically reduce training and prediction time for GBMs, making them ideal for large datasets or scenarios where quick model iteration is crucial.
 
Financial institutions deal with massive datasets and complex models that demand high computational power. With AMD ROCm™ and GPU-accelerated libraries like LightGBM and ThunderGBM, we can significantly reduce training time while improving model performance. In this post, we’ll explore two practical use cases and demonstrate the performance gains for each. For more information on the individual libraries, as well as the AMD ROCm Finance Toolkit they are a part of, please have a look at the [documentation](https://rocm.docs.amd.com/projects/rocm-finance/en/latest/).



### Installation Instructions

For instructions on how to install the individual components on AMD MI300X platforms, please refer to the installation instructions for [LightGBM](https://rocm.docs.amd.com/projects/lightgbm/en/latest/) and [ThunderGBM](https://rocm.docs.amd.com/projects/thundergbm/en/latest/).

### Common Benchmark Metrics
Various metrics are used to determine how well a GBM performs. Some commonly used metrics are:

- **AUC (Area Under the ROC Curve)**: AUC measures how well the model can separate fraudulent vs. legitimate transactions. A score of  **1.0**  is perfect, while **0.5**  is random guessing. For fraud detection, values above  **0.85**  are considered strong.
- **Accuracy**: Measures how often the model is correct. However, since fraud is only a tiny portion of all transactions, accuracy can be misleading (predicting all transactions as non-fraud gives high accuracy but zero fraud detection ability).
- **Precision**: Answers the question:  _"Of all transactions the model flagged as fraud, how many were actually fraud?"_  High precision means fewer false alarms.
- **Recall**: Answers the question:  _"Of all real fraud cases, how many did the model detect?"_  High recall is crucial because missed fraud means financial losses.

### Use Case 1: Predicting Loan Defaults with LightGBM on ROCm
In this example, we will demonstrate a use case where we build a prediction system using the LightGBM library with a binary classification model. 
This dataset is from a classic [Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/data) focused on a real-world problem: predicting whether a loan applicant will be able to repay their loan or will default. 
The dataset is particularly interesting because it's not a single flat file; it's a relational set of tables that mimics how data is often stored in real business environments.
#### Key Characteristics:
- **Primary Goal**: Predict the TARGET variable (0 for loan repaid, 1 for loan not repaid) for each SK_ID_CURR (loan application ID).

- **Structure**: The dataset consists of multiple files that need to be joined and aggregated. The main files include:
    - _application_train.csv / application_test.csv_: The main training and testing data with one row per loan application. Contains demographic and loan-specific information.
    - _bureau.csv_: Information about applicants' previous credits from other financial institutions.
    - _previous_application.csv_: Information about applicants' previous loan applications with Home Credit.
    - Several other files (_POS_CASH_balance.csv, credit_card_balance.csv, etc._) detail the history of previous loans.
- **Size and Scale**:
    - _application_train.csv_ contains 307,511 rows and 122 columns.
    - The entire unzipped dataset is over 1.5 GB.
    - The total number of features can easily expand into the thousands after feature engineering.
#### Data Loading and Preprocessing
In this section, we load the dataset and show how categorical features are encoded, missing values are handled, and how the data is split into training and validation sets.
```python
def train_lightgbm(data_path="application_train.csv",
                   num_leaves=2, learning_rate=0.03,
                   num_rounds=1000, max_depth=10,
                   min_data_in_leaf=50, compare_cpu=False):

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return f"❌ Error loading dataset: {e}", None, None, None, None, None

    if "TARGET" not in df.columns or "SK_ID_CURR" not in df.columns:
        return "❌ Dataset must contain 'TARGET' and 'SK_ID_CURR' columns.", None, None, None, None, None

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])
    applicant_ids = X["SK_ID_CURR"]
    X = X.drop(columns=["SK_ID_CURR"])

    X = X.fillna(np.nan)
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.loc[:, X.nunique(dropna=False) > 1]

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, applicant_ids, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

```
#### Model Training
We train the LightGBM model on an AMD MI300X GPU and optionally compare it with CPU performance, using the following training parameters:
- 80/20 train-validation split.
- Early stopping after 50 rounds without improvement.
- Up to 1000 boosting rounds.
``` python
gpu_params = base_params.copy()
gpu_params.update({
    "device_type": "cuda",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_threads": 16,
})

start_gpu = time.time()
model_gpu = lgb.train(
    gpu_params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=int(num_rounds),
    callbacks=callbacks
)
gpu_time = time.time() - start_gpu

y_pred_proba = model_gpu.predict(X_val, num_iteration=model_gpu.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)
auc = roc_auc_score(y_val, y_pred_proba)
acc = accuracy_score(y_val, y_pred)
```
#### Results
The results of the LightGBM training on the GPU are summarized as follows:
| Model           | Accuracy   |    AUC    | Training Time (sec) |
|-----------------|:----------:|:---------:|:-------------------:|
| **LightGBM GPU**|  0.9196    |  0.7484   |       2.59          |


### Use Case 2: Detecting Credit Card Fraud with ThunderGBM on ROCm
In this example, we develop a ThunderGBM model on the [IEEE-CIS credit card fraud dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) to predict fraudulent transactions. This dataset was developed for the IEEE Computational Intelligence Society (IEEE-CIS) Kaggle competition to improve fraud detection systems. The dataset has the following characteristics.
#### Dataset Size:
- Train Transaction Data: ~590,000 rows
- Test Transaction Data: ~500,000 rows
- Identity Data: ~144,000 rows (train) and ~141,000 rows (test)
#### Data Loading and Preprocessing
The first step in our benchmark involves loading and preprocessing the IEEE-CIS fraud detection dataset. The dataset consists of transaction and identity data, which are merged and encoded to prepare for model training.
```python
def load_and_preprocess():
    if os.path.exists(CACHE_PATH):
        print("Loading cached preprocessed dataset…")
        return pd.read_pickle(CACHE_PATH)

    trans_path = "/root/train_transaction.csv"
    ident_path = "/root/train_identity.csv"

    if not os.path.exists(trans_path) or not os.path.exists(ident_path):
        raise FileNotFoundError("Missing IEEE-CIS dataset CSVs.")

    print("Loading transaction data…")
    train_transaction = pd.read_csv(trans_path, low_memory=False)

    print("Loading identity data…")
    train_identity = pd.read_csv(ident_path, low_memory=False)

    print("Merging datasets…")
    df = train_transaction.merge(train_identity, on="TransactionID", how="left")

    print("Encoding categorical features…")
    for col in df.select_dtypes(include=["object", "category"]):
        df[col], _ = pd.factorize(df[col], sort=False)

    df = df.fillna(0).astype(np.float32)

    print("Saving cached preprocessed dataset…")
    df.to_pickle(CACHE_PATH)
    return df
```
#### Model Training
ThunderGBM is a GPU-only implementation and there is no CPU implementation to be used as reference. To compare the accuracy and performance of the GPU model we use the LightGBM library in CPU mode. The training process involves setting up the models with specific parameters and measuring the training time, AUC, accuracy, precision, and recall.
#### LightGBM CPU Training
```python
xgb_model = xgb.XGBClassifier(
    tree_method="hist",
    n_estimators=int(n_estimators),
    max_depth=int(max_depth),
    learning_rate=float(learning_rate),
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=1,
    verbosity=0
)

cpu_start = time.time()
xgb_model.fit(X_train, y_train)
cpu_time = time.time() - cpu_start

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_labels = (xgb_probs > 0.5).astype(int)

cpu_auc = roc_auc_score(y_test, xgb_probs)
cpu_acc = accuracy_score(y_test, xgb_labels)
cpu_precision = precision_score(y_test, xgb_labels)
cpu_recall = recall_score(y_test, xgb_labels)
```
#### ThunderGBM GPU Training
```python
tgbm_model = TGBMClassifier(
    depth=int(max_depth),
    n_trees=int(n_estimators),
    learning_rate=float(learning_rate),
    objective="binary:logistic",
    lambda_tgbm=1.0,
    max_num_bin=128,
    n_parallel_trees=4,
    min_child_weight=4,
    column_sampling_rate=0.8,
    verbose=0,
    n_gpus=1
)

gpu_start = time.time()
tgbm_model.fit(X_train, y_train)
gpu_time = time.time() - gpu_start

gpu_preds = tgbm_model.predict(X_test)

gpu_auc = roc_auc_score(y_test, gpu_preds)
gpu_acc = accuracy_score(y_test, gpu_preds)
gpu_precision = precision_score(y_test, gpu_preds)
gpu_recall = recall_score(y_test, gpu_preds)
```
### Results
The results of the benchmark are summarized as follows:


| Model           | Runtime (sec) |    AUC    | Accuracy  | Precision | Recall   |
|-----------------|:-------------:|:---------:|:---------:|:---------:|:--------:|
| **LightGBM CPU**      |   17.641      |  0.89350  |  0.97356  |  0.77595  | 0.32554  |
| **ThunderGBM GPU**    |   12.150      |  0.72883  |  0.94627  |  0.31912  | 0.49532  |

GPU Speedup
_ThunderGBM GPU is  1.45× faster than LightGBM CPU._

### Resources
This blog demonstrates how to use gradient boosting libraries on MI300X GPUs for finance focused workloads. To explore more on what we can do in terms of UI and data visualization, please refer to the examples folder in the [ROCm Finance GitHub Repo](https://github.com/ROCm/rocm-finance) for features like  
* Have a user facing UI for data inputs using [gradio](https://www.gradio.app/)
* Advanced Plotting for input and output data visualization using [matplotlib](https://matplotlib.org/)

We are also planning to add more opensource gradient boosting libraries to our ROCM-Finance toolkit, so stay tuned for more updates on this topic.

## Summary

This benchmark demonstrates the significant performance improvements achieved by leveraging GPU acceleration for machine learning tasks. ThunderGBM on GPU outperforms LightGBM on CPU in terms of training time, while LightGBM on AMD ROCm GPU provides efficient and accurate predictions for loan repayment probabilities. These results highlight the potential of GPU-accelerated machine learning for large-scale data processing and real-time predictions.


## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
