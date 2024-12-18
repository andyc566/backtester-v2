import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_model(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Define features and target
    X = df[['HDD', 'CDD']]
    y = df['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return evaluation metrics and model details
    return {
        'Mean Squared Error': mse,
        'R-squared': r2,
        'Intercept': model.intercept_,
        'Coefficients': model.coef_.tolist()
    }

def detrend_price_series(df):
    # Define features and target
    X = df[['HDD', 'CDD']]
    y = df['Price']

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the trend
    trend = model.predict(X)

    # Detrend the series
    df['Detrended_Price'] = y - trend

    return df, model