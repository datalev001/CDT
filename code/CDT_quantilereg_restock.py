import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt

# Load data
data_df = pd.read_csv('trans_data.csv')
current_stock_df = pd.read_csv('current_stock.csv')
product_importance_df = pd.read_csv('product_importance.csv')

def process_product(data, product_name, lead_time=7, service_level_factor=1.645, quantiles=[0.25, 0.5, 0.75], plot=False):
    # Fetch product importance and current inventory level from the CSV files
    try:
        product_importance = product_importance_df.loc[product_importance_df['Description'] == product_name, 'importance'].values[0]
    except IndexError:
        product_importance = 1.0  # Default importance if not found in CSV

    try:
        current_inventory_level = current_stock_df.loc[current_stock_df['Description'] == product_name, 'current_stock'].values[0]
    except IndexError:
        current_inventory_level = 50  # Default inventory if not found in CSV

    # Filter data for the specific product
    product_data = data[data['Description'] == product_name].copy()
    product_data = product_data.set_index('trans_date').resample('D').sum().reset_index()  # Resample to daily
    product_data['day_of_week'] = product_data['trans_date'].dt.dayofweek
    product_data['month'] = product_data['trans_date'].dt.month
    product_data['lag_1'] = product_data['Quantity'].shift(1)  # AR lag 1 day
    product_data['lag_7'] = product_data['Quantity'].shift(7)  # AR lag 7 days
    product_data['ma_7'] = product_data['Quantity'].rolling(window=7).mean().shift(1)  # MA effect: 7-day moving average
    product_data.dropna(inplace=True)  # Drop NaN values due to lagging

    # Train-test split based on real data
    train = product_data[product_data['trans_date'] < '2011-12-01']
    test = product_data[(product_data['trans_date'] >= '2011-12-01') & (product_data['trans_date'] < '2011-12-09')]

    # Perform Quantile Regression for each quantile level
    predictions = pd.DataFrame({'trans_date': test['trans_date']})
    for q in quantiles:
        model = QuantReg(train['Quantity'], train[['day_of_week', 'month', 'lag_1', 'lag_7', 'ma_7']])
        res = model.fit(q=q)
        predictions[f'quantile_{int(q * 100)}'] = res.predict(test[['day_of_week', 'month', 'lag_1', 'lag_7', 'ma_7']])

    # Summarize quantile predictions for lead time demand and safety stock
    quantile_25 = predictions['quantile_25'].mean()
    quantile_50 = predictions['quantile_50'].mean()
    quantile_75 = predictions['quantile_75'].mean()

    # Lead Time Demand (D_L): Sum median demand over the lead time
    daily_median_demand = quantile_50
    lead_time_demand = daily_median_demand * lead_time

    # Safety Stock (SS): Adjusted using quantile-based variance
    demand_variance = quantile_75 - quantile_25
    safety_stock = service_level_factor * demand_variance

    # Utility Calculation (incorporating product importance)
    lambda_1, lambda_2, lambda_3 = 0.5, 1.0, 1.0
    expected_return = quantile_50
    variance = quantile_75 - quantile_25
    upside_potential = quantile_75 - quantile_50
    downside_risk = quantile_50 - quantile_25
    utility = (expected_return - (lambda_1 * variance) + (lambda_2 * upside_potential) - (lambda_3 * downside_risk)) * product_importance

    # Restocking Recommendation
    reorder_point = lead_time_demand + safety_stock
    restocking_amount = max(0, reorder_point - current_inventory_level)

    # Plot results if plot=True
    if plot:
        plt.figure(figsize=(14, 8))

        # Line plot of historical demand with quantile predictions
        plt.plot(product_data['trans_date'], product_data['Quantity'], label='Historical Demand', color='blue')
        plt.plot(predictions['trans_date'], predictions['quantile_25'], label='Quantile 25%', linestyle='--', color='orange')
        plt.plot(predictions['trans_date'], predictions['quantile_50'], label='Quantile 50% (Median)', linestyle='-', color='green')
        plt.plot(predictions['trans_date'], predictions['quantile_75'], label='Quantile 75%', linestyle='--', color='red')
        plt.axhline(y=reorder_point, color='purple', linestyle='-', label='Reorder Point')
        plt.axhline(y=current_inventory_level, color='black', linestyle=':', label='Current Inventory Level')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.title(f'Demand and Quantile Forecasting for {product_name}')
        plt.legend()
        plt.show()

        # Bar plot of quantile values for the product
        plt.figure(figsize=(8, 6))
        plt.bar(['Quantile 25%', 'Quantile 50% (Median)', 'Quantile 75%'], [quantile_25, quantile_50, quantile_75], color=['orange', 'green', 'red'])
        plt.title(f'Quantile Demand Levels for {product_name}')
        plt.ylabel('Quantity')
        plt.show()

        # Bar plot comparing current inventory vs reorder point
        plt.figure(figsize=(8, 6))
        plt.bar(['Current Inventory', 'Reorder Point'], [current_inventory_level, reorder_point], color=['grey', 'purple'])
        plt.title(f'Inventory vs Reorder Point for {product_name}')
        plt.ylabel('Quantity')
        plt.show()

    # Return the result as a dictionary
    return {
        'product_name': product_name,
        'utility': utility,
        'restocking_amount': restocking_amount,
        'current_inventory': current_inventory_level,
        'quantile_25_total': quantile_25,
        'quantile_50_total': quantile_50,
        'quantile_75_total': quantile_75,
        'lead_time_demand': lead_time_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point
    }


# Run the function for the first product in your data with plotting enabled
first_product = data_df['Description'].unique()[0]
first_product_result = process_product(data_df, first_product, lead_time=7, service_level_factor=1.645, plot=True)

# Display Results in descending order of utility
print("Product Utilities (Ranked):")
for result in sorted(results, key=lambda x: x['utility'], reverse=True):
    print(f"{result['product_name']}: Utility = {result['utility']:.2f}, Restocking Amount = {result['restocking_amount']:.2f}, Current Inventory = {result['current_inventory']}")
