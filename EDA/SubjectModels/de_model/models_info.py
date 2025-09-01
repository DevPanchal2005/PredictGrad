models = [
  # Multiple Linear Regression (MSE loss)
  {
    "Model": "Multiple Linear Regression (MSE loss)",
    "Approach": "multivariate regression + 5-Fold cv + one-hot encoding",
    "MAE": 7.5616,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)


# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize linear regression model
model = LinearRegression()

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores across folds
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

# Convert to positive MAE values
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: Multiple Linear Regression (MSE loss)")
print("Approach: multivariate regression + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")
"""
  },
  # Multiple Linear Regression (MSE loss + High VIF columns dropped)
  {
    "Model": "Multiple Linear Regression (MSE loss + High VIF columns dropped)",
    "Approach": "Multivariate regression + 5-Fold CV + one-hot encoding",
    "MAE": 7.624,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)
# Without dropping High VIF columns: MAE: 6.6619

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize linear regression model
model = LinearRegression()

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores across folds
neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

# Convert to positive MAE values
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: Multiple Linear Regression (MSE loss + High VIF columns dropped)")
print("Approach: Multivariate regression + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Quantile Regression (MAE loss)
  {
    "Model": "Quantile Regression (MAE loss)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.5496,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Add intercept manually
X = sm.add_constant(X)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

# Fit Quantile Regression (MAE = q=0.5) on each fold
for train_index, test_index in kf.split(X):
    # Ensure input is float type to prevent dtype=object errors
    X_train = X.iloc[train_index].astype(float)
    X_test = X.iloc[test_index].astype(float)
    y_train = y.iloc[train_index].astype(float)
    y_test = y.iloc[test_index].astype(float)

    # Fit Quantile Regression model (q=0.5 corresponds to MAE minimization)
    model = sm.QuantReg(y_train, X_train)
    result = model.fit(q=0.5)

    # Predict and calculate fold MAE
    preds = result.predict(X_test)
    fold_mae = np.mean(np.abs(y_test - preds))
    mae_scores.append(fold_mae)

mean_mae = np.mean(mae_scores)

# Print and log
print("Model: Quantile Regression (MAE loss)")
print("Approach: q=0.5 + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Quantile Regression (MAE loss High VIF columns dropped)
  {
    "Model": "Quantile Regression (MAE loss High VIF columns dropped)",
    "Approach": "q=0.5 + 5-Fold CV + one-hot encoding",
    "MAE": 7.7069,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")
# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Add intercept manually
X = sm.add_constant(X)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

# Fit Quantile Regression (MAE = q=0.5) on each fold
for train_index, test_index in kf.split(X):
    # Ensure input is float type to prevent dtype=object errors
    X_train = X.iloc[train_index].astype(float)
    X_test = X.iloc[test_index].astype(float)
    y_train = y.iloc[train_index].astype(float)
    y_test = y.iloc[test_index].astype(float)

    # Fit Quantile Regression model (q=0.5 corresponds to MAE minimization)
    model = sm.QuantReg(y_train, X_train)
    result = model.fit(q=0.5)

    # Predict and calculate fold MAE
    preds = result.predict(X_test)
    fold_mae = np.mean(np.abs(y_test - preds))
    mae_scores.append(fold_mae)

mean_mae = np.mean(mae_scores)

# Print and log
print("Model: Quantile Regression (MAE loss, High VIF columns dropped)")
print("Approach: q=0.5 + 5-Fold cv + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 2)
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2",
    "MAE": 28.8104,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 2)
polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 2)")
print(
    "Approach: Full-feature polynomial regression (degree 2) with 5-Fold CV and one-hot encoding"
)
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 2) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 2)",
    "Approach": "5-Fold CV + one-hot encoding + degree 2 + high VIF columns dropped",
    "MAE": 33.3695,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 2)
polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 2)")
print("Approach: 5-Fold CV + one-hot encoding + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 3)
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3",
    "MAE": 18.0693,
    "Code": """
# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 3)
polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 3)")
print("Approach: Full-feature polynomial regression + 5-Fold CV + one-hot encoding")
print(f"MAE: {mean_mae:.4f}")"""
  },
  # Polynomial Regression (Order 3) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 3)",
    "Approach": "5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped",
    "MAE": 18.8222,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")


# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 3)
polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 3)")
print("Approach: 5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 3)",
            "Approach": "5-Fold CV + one-hot encoding + degree 3 + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Polynomial Regression (Order 4) 
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4",
    "MAE": 16.7219,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 4)
polyreg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 4)")
print("Approach: 5-Fold CV + one-hot encoding + degree 4")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 4)",
            "Approach": "5-Fold CV + one-hot encoding + degree 4",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Polynomial Regression (Order 4) (high VIF columns dropped)
  {
    "Model": "Polynomial Regression (Order 4)",
    "Approach": "5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped",
    "MAE": 17.5248,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# drop columns with too high VIF
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]

# Drop columns, ignoring those not found
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define target and feature columns
target_col = "DE Theory"

# All remaining columns except target are used as features
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Initialize polynomial regression (order 4)
polyreg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute Negative MAE scores
neg_mae_scores = cross_val_score(
    polyreg, X, y, cv=kf, scoring="neg_mean_absolute_error"
)

# Convert to positive MAE
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Polynomial Regression (Order 4)")
print("Approach: 5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped")
print(f"MAE: {mean_mae:.4f}")

# Store results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Polynomial Regression (Order 4)",
            "Approach": "5-Fold CV + one-hot encoding + degree 4 + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Support Vector Regression (RBF)
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler",
    "MAE": 8.4182,
    "Code": """# One-hot encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)


# Define features and target
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Build pipeline: Standardize -> SVR
svr_pipeline = make_pipeline(
    StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
)

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    svr_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Support Vector Regression (RBF)")
print("Approach: 5-Fold CV + one-hot encoding + StandardScaler + RBF")
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Support Vector Regression (RBF)",
            "Approach": "5-Fold CV + one-hot encoding + StandardScaler",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Support Vector Regression (RBF) (high VIF columns dropped)
  {
    "Model": "Support Vector Regression (RBF)",
    "Approach": "5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped",
    "MAE": 8.5306,
    "Code": """# One-hot encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Drop high-VIF columns
columns_to_drop = [
    "Math-1 Theory",
    "DBMS Theory",
    "Sem 2 Percentage",
    "Sem 1 Percentage",
]
df_encoded = df_encoded.drop(columns=columns_to_drop, errors="ignore")

# Define features and target
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Build pipeline: Standardize -> SVR
svr_pipeline = make_pipeline(
    StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
)

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    svr_pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Support Vector Regression (RBF)")
print(
    "Approach: 5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Support Vector Regression (RBF)",
            "Approach": "5-Fold CV + one-hot encoding + StandardScaler + RBF kernel + high VIF columns dropped",
            "MAE": round(mean_mae, 4),
        }
    ]
)
results_df.to_csv(
    "model_results_log.csv",
    mode="a",
    header=not pd.io.common.file_exists("model_results_log.csv"),
    index=False,
)"""
  },
  # Random Forest Regressor
  {
    "Model": "Random Forest Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.0474,
    "Code": """# One-hot encode categorical columns and drop the first column of each
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Religion", "Branch", "Section-1", "Section-2", "Section-3"],
    drop_first=True,
)

# Didn't drop columns with high internal correlation because tree based structures handel them well

# Define target and feature columns
target_col = "DE Theory"
feature_cols = [col for col in df_encoded.columns if col != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

# Define model pipeline (no preprocessor needed since categorical columns are already encoded)
model = Pipeline(
    steps=[
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Use 5-Fold CV with negative MAE
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Suppress specific sklearn warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    neg_mae_scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_absolute_error"
    )

mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results to terminal
print("Model: Random Forest Regressor")
print("Approach: Full-feature regression with 5-Fold CV and OneHotEncoding")
print(f"MAE: {mean_mae:.4f}")

# Log results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor",
            "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, 'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}",
    "MAE": 7.8887,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                max_depth=20,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, "
    "'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 200, 'regressor__min_samples_split': 2, "
            "'regressor__min_samples_leaf': 2, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 20}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}",
    "MAE": 7.9547,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline with updated hyperparameters
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features=None,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, "
    "'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 1000, 'regressor__min_samples_split': 5, "
            "'regressor__min_samples_leaf': 4, 'regressor__max_features': None, 'regressor__max_depth': None}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}",
    "MAE": 7.8615,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline with the specified tuned hyperparameters
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=None,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Output
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, "
    "'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, "
            "'regressor__min_samples_leaf': 3, 'regressor__max_features': 0.5, 'regressor__max_depth': None}",
            "MAE": round(mean_mae, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 4, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 30}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 10, 'regressor__min_samples_leaf': 4, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 30}",
    "MAE": 7.8275,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical columns to encode
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]


# Preprocessing pipeline for encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Define parameter grid for RandomizedSearchCV
param_distributions = {
    "regressor__n_estimators": [100, 200, 500, 1000],
    "regressor__max_depth": [10, 20, 30, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
    "regressor__max_features": ["sqrt", "log2", None],  # Removed 'auto'
}


# Use 5-Fold CV for tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    scoring="neg_mean_absolute_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
)

# Fit the RandomizedSearchCV to the data
random_search.fit(X, y)

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_  # Convert back from neg MAE to MAE

# Print results to terminal
print("Model: Random Forest Regressor(tuned)")
print("Parameters:", best_params)
print(f"MAE: {best_score:.4f}")

# Log results to CSV
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": best_params,
            "MAE": round(best_score, 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 0.5, 'max_depth': 30}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 0.5, 'max_depth': 30}",
    "MAE": 7.8671,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric columns
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Define the tuned Random Forest model
tuned_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=10,
                min_samples_leaf=3,
                max_features=0.5,
                max_depth=30,
                random_state=42,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mae_scores = cross_val_score(
    tuned_model, X, y, cv=kf, scoring="neg_mean_absolute_error"
)
mae_scores = -neg_mae_scores
mean_mae = np.mean(mae_scores)

# Print results
print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, "
    "'max_features': 0.5, 'max_depth': 30}"
)
print(f"MAE: {mean_mae:.4f}")

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, "
            "'max_features': 0.5, 'max_depth': 30}",
            "MAE": round(mean_mae, 4),
        }
    ]
)
log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # Random Forest Regressor (Tuned) {'regressor__n_estimators': 500, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 10}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'regressor__n_estimators': 500, 'regressor__min_samples_split': 5, 'regressor__min_samples_leaf': 1, 'regressor__max_features': 'sqrt', 'regressor__max_depth': 10}",
    "MAE": 7.812,
    "Code": """# Handle outliers in target using IQR
target_col = "DE Theory"
Q1, Q3 = df[target_col].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[~((df[target_col] < Q1 - 1.5 * IQR) | (df[target_col] > Q3 + 1.5 * IQR))]

# Define target and features
X = df.drop(columns=[target_col])
y = df[target_col]

# Categorical and numeric features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Full pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Hyperparameter tuning setup
param_distributions = {
    "regressor__n_estimators": [100, 200, 500],
    "regressor__max_depth": [10, 20, 30, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 3],
    "regressor__max_features": ["sqrt", 0.5, None],
}

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=kf,
    random_state=42,
    n_jobs=-1,
)

# Fit model
random_search.fit(X, y)

# Best model and results
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

# Feature importances
feature_importances = best_model.named_steps["regressor"].feature_importances_
feature_names = (
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
    .tolist()
    + numeric_cols
)
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
)
print(
    "Feature Importances:\n",
    importance_df.sort_values(by="Importance", ascending=False),
)

# Log results
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": best_params,
            "MAE": round(best_score, 4),
        }
    ]
)
log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)

# Output summary
print("Model: Random Forest Regressor (Tuned)")
print("Approach:", best_params)
print(f"MAE: {best_score:.4f}")"""
  },
  # Random Forest Regressor (Tuned) {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}
  {
    "Model": "Random Forest Regressor (Tuned)",
    "Approach": "{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}",
    "MAE": 7.8142,
    "Code": """# Define target and features
target_col = "DE Theory"
X = df.drop(columns=[target_col])
y = df[target_col]


# Custom Winsorizer
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        # Must specify axis=1 since our lower/upper bounds have columns as the index
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


# Features
categorical_cols = [
    "Gender",
    "Religion",
    "Branch",
    "Section-1",
    "Section-2",
    "Section-3",
]

# Use top 25 features importance list
top_features = [
    "Sem 2 Percentage",
    "Math-2 Theory",
    "Data Structures using Java Theory",
    "Fundamental of Electronics and Electrical Theory",
    "Sem 1 Percentage",
    "Physics Theory",
    "DBMS Theory",
    "Math-1 Theory",
    "Software Engineering Theory",
    "DBMS Practical",
    "Java-1 Theory",
    "Java-2 Theory",
    "Fundamental of Electronics and Electrical Practical",
    "Environmental Science Theory",
    "Data Structures using Java Practical",
    "Java-2 Attendance",
    "Data Structures using Java Attendance",
    "Software Engineering Attendance",
    "Fundamental of Electronics and Electrical Attendance",
    "Roll-1",
    "Java-1 Attendance",
    "Math-1 Attendance",
    "Math-2 Attendance",
    "Computer Workshop Practical",
    "Physics Attendance",
]

# Add relevant categorical columns (one-hot encoding will handle dummy drop)
top_features += categorical_cols
X = X[top_features]

# Identify numeric columns (excluding categoricals)
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline([("winsor", Winsorizer()), ("scaler", StandardScaler())]),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# Tuned Random Forest Model
model = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=1000,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.3,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Cross-Validation MAE
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = -1 * cross_val_score(
    model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
)

print("Model: Random Forest Regressor (Tuned)")
print(
    "Parameters: {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}"
)
print(f"Mean MAE: {mae_scores.mean():.4f}")

# --- Log Results to CSV ---
results_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest Regressor (Tuned)",
            "Approach": "{'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.3, 'max_depth': 15}",
            "MAE": round(mae_scores.mean(), 4),
        }
    ]
)

log_file = "model_results_log.csv"
results_df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)"""
  },
  # XGBoost Regressor
  {
    "Model": "XGBoost Regressor",
    "Approach": "Full-feature regression + OneHotEncoding + 5-Fold CV",
    "MAE": 8.7712,
    "Code": ""
  },
  {
    "Model": "XGBoost Regressor",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.8, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.8})",
    "MAE": 7.8803,
    "Code": ""
  },
  {
    "Model": "XGBoost Regressor(Tuned)",
    "Approach": "Tuned (Best Params: {'regressor__colsample_bytree': 0.9, 'regressor__learning_rate': 0.05, 'regressor__max_depth': 3, 'regressor__n_estimators': 100, 'regressor__subsample': 0.9})",
    "MAE": 7.9463,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor",
    "Approach": "Full-feature regression with 5-Fold CV and OneHotEncoding",
    "MAE": 8.3525,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.9, 'regressor__num_leaves': 70, 'regressor__n_estimators': 100, 'regressor__max_depth': -1, 'regressor__learning_rate': 0.03, 'regressor__colsample_bytree': 1.0})",
    "MAE": 7.9638,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with RandomizedSearchCV (params: {'regressor__subsample': 0.8, 'regressor__num_leaves': 70, 'regressor__n_estimators': 500, 'regressor__min_child_samples': 30, 'regressor__max_depth': 3, 'regressor__learning_rate': 0.01, 'regressor__colsample_bytree': 0.7})",
    "MAE": 7.8892,
    "Code": ""
  },
  {
    "Model": "LightGBM Regressor (Tuned)",
    "Approach": "Tuned with BayesSearchCV (params: OrderedDict({'regressor__colsample_bytree': 1.0, 'regressor__learning_rate': 0.012614141235943423, 'regressor__max_depth': 3, 'regressor__min_child_samples': 44, 'regressor__n_estimators': 540, 'regressor__num_leaves': 20, 'regressor__reg_alpha': 0.0, 'regressor__reg_lambda': 0.22975045403226968, 'regressor__subsample': 1.0}))",
    "MAE": 7.8728,
    "Code": ""
  },
  {
    "Model": "Ridge Regression",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization",
    "MAE": 7.5447,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Tuned alpha=79.0604 using GridSearchCV",
    "MAE": 7.4468,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Full-feature regression with 5-Fold CV and Regularization alpha: 100.0",
    "MAE": 7.448,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection and polynomial features with 5-Fold CV Alpha: 10.0 Number of Features: 10",
    "MAE": 7.7575,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Tuned regression with 5-Fold CV Best Alpha: 112.8838",
    "MAE": 7.4505,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (Tuned)",
    "Approach": "Feature selection + polynomial features + 5-Fold CV + Best Alpha: 0.01 + Number of Features: 12",
    "MAE": 7.739,
    "Code": ""
  },
  {
    "Model": "Ridge Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4408,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + 5-Fold CV and L1+L2 Regularization",
    "MAE": 7.4618,
    "Code": ""
  },
  {
    "Model": "Lasso Regression (tuned)",
    "Approach": "Tuned regression + 5-Fold CV Best Alpha: 0.07847599703514611",
    "MAE": 7.4711,
    "Code": ""
  },
  {
    "Model": "Lasso Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha selected via two-stage grid search",
    "MAE": 7.4746,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression",
    "Approach": "Full-feature regression + Repeated 5-Fold CV + GridSearch on alpha + L1 ratio",
    "MAE": 7.4457,
    "Code": ""
  },
  {
    "Model": "ElasticNet Regression (tuned)",
    "Approach": "Full-feature regression with Repeated 5-Fold CV and Regularization, alpha and L1 ratio selected via grid search",
    "MAE": 7.4457,
    "Code": ""
  },
  {
    "Model": "ExtraTrees Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.7777,
    "Code": ""
  },
  {
    "Model": "HistGradientBoosting Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 7.9384,
    "Code": ""
  },
  {
    "Model": "NGBoost Regressor",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + GridSearch",
    "MAE": 8.0045,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Ridge meta-model",
    "MAE": 7.5258,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + XGBoost + LightGBM)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + CatBoost meta-model",
    "MAE": "",
    "Code": ""
  },
  {
    "Model": "Stacking Regressor (Ridge + ElasticNet + XGBoost + LightGBM + CatBoost)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + Stacking",
    "MAE": 7.5193,
    "Code": ""
  },
  {
    "Model": "StackNet GradientBoosting Stacking (Layer 1 + Layer 2)",
    "Approach": "Layer 1: Ridge + Lasso + XGBoost; Layer 2: GradientBoostingRegressor; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 8.1737,
    "Code": ""
  },
  {
    "Model": "Voting Ensemble (Tree-Based Only)",
    "Approach": "RandomForest + ExtraTrees + LightGBM + XGBoost; Weighted Voting; OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.9487,
    "Code": ""
  },
  {
    "Model": "Stacked Regressor (Ridge + ElasticNet + RandomForest + XGBoost)",
    "Approach": "Diverse Feature Sets + OneHot + RobustScaler + Repeated 5-Fold CV + Linear meta-model",
    "MAE": 7.5496,
    "Code": ""
  },
  {
    "Model": "Bootstrap Aggregated XGBoost",
    "Approach": "10 Bootstrapped XGBoost Models + Averaged Predictions + OneHot + RobustScaler + Repeated 5-Fold CV",
    "MAE": 7.859,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Weighted Voting",
    "MAE": 7.4714,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + Random Forest)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.5819,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Manual Weights",
    "MAE": 7.4702,
    "Code": ""
  },
  {
    "Model": "Voting Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV",
    "MAE": 7.1206,
    "Code": ""
  },
  {
    "Model": "Stacking Regressor (Ridge + Lasso + ElasticNet)",
    "Approach": "OneHot + RobustScaler + Repeated 5-Fold CV + BayesSearchCV + Stacking",
    "MAE": 7.4405,
    "Code": ""
  },
  {
    "Model": "Full Ensemble (Linear + Tree Models)",
    "Approach": "OneHot + RobustScaler + KFold CV + Stacking",
    "MAE": 7.5555,
    "Code": ""
  }
]
