import nltk
nltk.download('punkt_tab')

import pandas as pd
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Reload the file path
salary_data = pd.read_csv('salary_data_processed.csv', encoding='latin1')

def calculate_skill_level(requirements):
    if pd.isna(requirements) or requirements.lower() == "not found":
        return 0
    sentences = sent_tokenize(requirements)
    return len(sentences)

salary_data['skill_level'] = salary_data['requirements'].apply(calculate_skill_level)


def convert_to_monthly_salary(salary_str):
    if pd.isna(salary_str):
        return None
    usd_rate = 15000  # Approximate USD to IDR conversion rate
    days_per_month = 22  # Assuming 22 working days in a month
    hours_per_day = 8    # Assuming 8 working hours per day
    months_per_year = 12

    try:
        # If salary is in USD
        if "USD" in salary_str:
            match = re.search(r"\$(\d+(?:,\d+)?)(?:.*?(hour|day|year))?", salary_str.replace(",", ""))
            if match:
                usd_value = float(match.group(1))
                unit = match.group(2)
                if unit == "hour":
                    idr_value = usd_value * usd_rate * hours_per_day * days_per_month
                elif unit == "day":
                    idr_value = usd_value * usd_rate * days_per_month
                elif unit == "year":
                    idr_value = (usd_value * usd_rate) / months_per_year
                else:  # Assume monthly if no unit specified
                    idr_value = usd_value * usd_rate
                return idr_value

        # If salary is in IDR
        elif "Rp" in salary_str:
            match = re.search(r"(\d+(?:,\d+)?)(?:.*?(hour|day|year))?", salary_str.replace(".", "").replace(",", ""))
            if match:
                idr_value = int(match.group(1))
                unit = match.group(2)
                if unit == "hour":
                    idr_value = idr_value * hours_per_day * days_per_month
                elif unit == "day":
                    idr_value = idr_value * days_per_month
                elif unit == "year":
                    idr_value = idr_value / months_per_year
                # Already monthly if no unit specified
                return idr_value
    except:
        return None
    return None

# Apply the function to the 'salary' column to standardize all salaries to monthly
salary_data['adjusted_monthly_salary_idr'] = salary_data['salary'].apply(convert_to_monthly_salary)


# Feature Engineering (One-Hot Encoding)
salary_data['text_features'] = salary_data['requirements'].fillna("") + " " + salary_data['min_education_level'].fillna("")

education_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_education = education_encoder.fit_transform(salary_data[['min_education_level']])
encoded_df = pd.DataFrame(encoded_education, columns=education_encoder.get_feature_names_out(['min_education_level']))
salary_data = pd.concat([salary_data, encoded_df], axis=1)

# Updated Preprocessor
numeric_features = ['skill_level'] + list(encoded_df.columns)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('text', TfidfVectorizer(max_features=500, ngram_range=(1, 2)), 'text_features'),
    ])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Data Preparation
salary_data = salary_data.dropna(subset=['adjusted_monthly_salary_idr'])
X = salary_data[['text_features', 'skill_level'] + list(encoded_df.columns)]
y = salary_data['adjusted_monthly_salary_idr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_dist = {
    'regressor__max_depth': [5, 10, 20, 30],
    'regressor__min_samples_split': [5, 10, 20],
    'regressor__min_samples_leaf': [2, 5],
    'preprocess__text__max_features': [200, 500],
    'preprocess__text__ngram_range': [(1, 2)],
    'preprocess__text__ngram_range': [(1, 1), (1, 2)]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    verbose=1,
    random_state=42
)

# Fit and Evaluate
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Best Model R²: {r2:.2f}")
print(f"Best Model RMSE: {rmse:.2f}")