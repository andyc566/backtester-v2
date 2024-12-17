import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to de-weather a price series
def deweather_price_series(prices, weather_data, weather_features):
    # Merge the data on date
    data = pd.merge(prices, weather_data, on='date')

    # Define X and y for regression
    X = data[weather_features]
    y = data['price']

    # Train the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate weather effect
    weather_effect = model.predict(X)

    # De-weathered prices
    data['deweathered_price'] = data['price'] - (weather_effect - model.intercept_)

    return data[['date', 'price', 'deweathered_price']] 

# Example usage:
# prices = pd.DataFrame({'date': [...], 'price': [...]})
# weather_data = pd.DataFrame({'date': [...], 'temperature': [...], 'rainfall': [...]})
# deweathered = deweather_price_series(prices, weather_data, ['temperature', 'rainfall'])
# print(deweathered)
