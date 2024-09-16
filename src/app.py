import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load the CSV file
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv", sep=",")

# Display the first few rows of the DataFrame to ensure it's loaded correctly
df.head()

# List of categorical columns to one-hot encode
categorical_columns = ["sex", "smoker", "region"]

# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# List of numerical columns (after encoding)
num_variables = df_encoded.columns  # After encoding, all columns are included

fig, axis = plt.subplots(2, 3, figsize=(15, 10))  # Increase figure size for better readability

# Histogram for 'sex' (categorical data)
sns.histplot(ax=axis[0, 0], data=df, x="sex", discrete=True, shrink=0.8)
axis[0, 0].set_title("Sex Distribution")
axis[0, 0].set_xlabel("Sex")
axis[0, 0].set_ylabel("Count")
axis[0, 0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

# Histogram for 'smoker' (categorical data)
sns.histplot(ax=axis[0, 1], data=df, x="smoker", discrete=True, shrink=0.8)
axis[0, 1].set_title("Smoker Distribution")
axis[0, 1].set_xlabel("Smoker")
axis[0, 1].set_ylabel(None)  # Remove y-label to reduce clutter
axis[0, 1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

# Histogram for 'region' (categorical data)
sns.histplot(ax=axis[0, 2], data=df, x="region", discrete=True, shrink=0.8)
axis[0, 2].set_title("Region Distribution")
axis[0, 2].set_xlabel("Region")
axis[0, 2].set_ylabel(None)  # Remove y-label to reduce clutter
axis[0, 2].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

# Histogram for 'age' (numerical feature)
sns.histplot(ax=axis[1, 0], data=df, x="age", bins=50, kde=False)
axis[1, 0].set_title("Age Distribution")
axis[1, 0].set_xlabel("Age")
axis[1, 0].set_ylabel("Count")

# Histogram for 'bmi' (numerical feature)
sns.histplot(ax=axis[1, 1], data=df, x="bmi", bins=30, kde=False)
axis[1, 1].set_title("BMI Distribution")
axis[1, 1].set_xlabel("BMI")
axis[1, 1].set_ylabel(None)  # Remove y-label to reduce clutter

# Histogram for 'children' (numerical feature)
sns.histplot(ax=axis[1, 2], data=df, x="children", bins=30, kde=False)
axis[1, 2].set_title("Number of Children Distribution")
axis[1, 2].set_xlabel("Number of Children")
axis[1, 2].set_ylabel(None)  # Remove y-label to reduce clutter

# Adjust the layout to reduce overlap and improve spacing
plt.tight_layout()

# Show the plot
plt.show()

sns.pairplot(df)
plt.title('Pair Plot of Features')
plt.show()

correlation_matrix = df_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

x_columns = ["smoker_yes", "age", "bmi"] 
y_column = "charges"                     

total_data = df_encoded.copy()

# Create subplots
fig, axis = plt.subplots(len(x_columns) + 1, 1, figsize=(10, 15))  # Create more space for multiple plots

# Plot regression plots for each x_column
for i, x_column in enumerate(x_columns):
    sns.regplot(ax=axis[i], data=total_data, x=x_column, y=y_column)
    axis[i].set_title(f'Regression Plot of {y_column} vs {x_column}')
    axis[i].set_xlabel(x_column)
    axis[i].set_ylabel(y_column)

# Heatmap of correlation matrix for selected columns
selected_columns = x_columns + [y_column]
corr_matrix = total_data[selected_columns].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=axis[len(x_columns)], cbar=False, cmap='coolwarm')
axis[len(x_columns)].set_title('Correlation Heatmap')

# Adjust the layout to reduce overlap and improve spacing
plt.tight_layout()

# Show the plot
plt.show()

# The target variable is: 'charges'
target_column = 'charges'
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.1, 
                                                    random_state=42)

# Feature selection using SelectKBest with f_regression (for regression tasks)
selection_model = SelectKBest(f_regression, k=5)
selection_model.fit(X_train, y_train)

# Get boolean mask of selected features
support_mask = selection_model.get_support()

# Get the selected column names from the original DataFrame
selected_columns = X_train.columns[support_mask]

# Transform the train and test sets to retain only selected features
X_train = pd.DataFrame(selection_model.transform(X_train), columns=selected_columns, index=X_train.index)
X_test = pd.DataFrame(selection_model.transform(X_test), columns=selected_columns, index=X_test.index)

# Display the first few rows of selected features in the training set
X_train.head()

X_test.head()

# Scale the numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_encoded)

# Create DataFrame for scaled features
df_scal = pd.DataFrame(scaled_features, index=df.index, columns=num_variables)

# Display the first few rows of the scaled DataFrame
df_scal.head()

# Save the features (X_train and X_test) to CSV files
X_train.to_csv("../Data/medical_insurance_cost_train.csv", index=False)
X_test.to_csv("../Data/medical_insurance_cost_test.csv", index=False)

# Save the target variables (y_train and y_test) to CSV files
y_train.to_csv("../Data/medical_insurance_cost_train_target.csv", index=False)
y_test.to_csv("../Data/medical_insurance_cost_test_target.csv", index=False)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept (a): {model.intercept_}")
print(f"Coefficients (b): {model.coef_}")

y_pred = model.predict(X_test)
y_pred

# Create subplots for all selected features
fig, axis = plt.subplots(1, 3, figsize=(18, 6))  # 3 columns for age, bmi, and smoker_yes

# Plot 1: Charges vs Age
sns.scatterplot(ax=axis[0], data=test_data, x='age', y='charges', label='Actual Charges')
sns.lineplot(ax=axis[0], x=test_data['age'], y=regression_age(test_data['age']), color='red', label='Regression Line')
axis[0].set_title('Actual vs Predicted Charges (Age)')
axis[0].set_xlabel('Age')
axis[0].set_ylabel('Charges')

# Plot 2: Charges vs BMI
sns.scatterplot(ax=axis[1], data=test_data, x='bmi', y='charges', label='Actual Charges')
sns.lineplot(ax=axis[1], x=test_data['bmi'], y=regression_bmi(test_data['bmi']), color='red', label='Regression Line')
axis[1].set_title('Actual vs Predicted Charges (BMI)')
axis[1].set_xlabel('BMI')
axis[1].set_ylabel('Charges')

# Plot 3: Charges vs Smoker (Smoker_yes as 1 or 0)
sns.scatterplot(ax=axis[2], data=test_data, x='smoker_yes', y='charges', label='Actual Charges')
sns.lineplot(ax=axis[2], x=test_data['smoker_yes'], y=regression_smoker_yes(test_data['smoker_yes']), color='red', label='Regression Line')
axis[2].set_title('Actual vs Predicted Charges (Smoker)')
axis[2].set_xlabel('Smoker (1 = Yes, 0 = No)')
axis[2].set_ylabel('Charges')

# Adjust layout
plt.tight_layout()
plt.show()

