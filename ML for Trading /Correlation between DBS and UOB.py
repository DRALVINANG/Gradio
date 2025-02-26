# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import gradio as gr

# Function to download stock data and perform linear regression
def stock_analysis(dbs_price, selected_stock):
    # Define stock tickers and their names
    stock_tickers = {
        "DBS": 'D05.SI',
        "UOB": 'U11.SI',
        "OCBC": 'O39.SI',
        "Singtel": 'Z74.SI',
        "Standard Chartered": 'STAN.L',
        "HSBC": 'HSBC.L'
    }
    
    # Get the corresponding stock symbol
    stock_symbol = stock_tickers.get(selected_stock, 'D05.SI')

    start_date = '2020-01-01'

    # Download stock data
    data1 = yf.download(stock_tickers["DBS"], start=start_date)  # DBS
    data2 = yf.download(stock_symbol, start=start_date)  # Selected stock

    data1.columns = data1.columns.droplevel(level=1)
    data2.columns = data2.columns.droplevel(level=1)

    # Merge both stock prices into one dataframe
    data = pd.merge(data1.Close, data2.Close, left_index=True, right_index=True)
    data.rename(columns={'Close_x': 'DBS Close Price', 'Close_y': f'{selected_stock} Close Price'}, inplace=True)

    # Linear Regression Model
    y = data[f'{selected_stock} Close Price'].values.reshape(-1, 1)
    X = data['DBS Close Price'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    # Predicted value based on DBS price
    predicted_value = model.predict([[dbs_price]])

    # Linear regression equation
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f"y = {slope:.2f}x + {intercept:.2f}"

    # Plot the scatter plot and regression line
    plt.figure(figsize=(12, 5))
    plt.scatter(data['DBS Close Price'], data[f'{selected_stock} Close Price'], label='Data Points')
    plt.plot(data['DBS Close Price'], model.intercept_[0] + model.coef_[0][0] * data['DBS Close Price'], color='red', label='Regression Line')
    plt.xlabel('DBS Price')
    plt.ylabel(f'{selected_stock} Price')
    plt.title(f'Linear Regression: DBS vs {selected_stock} Stock Prices')
    plt.legend()

    # Save the plot to a file
    plot_filename = "/tmp/stock_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    # Return the prediction, plot filename, and the linear regression equation
    return f"Predicted {selected_stock} Close Price for DBS price of {dbs_price}: {predicted_value[0][0]:.2f}", equation, plot_filename

# Create the Gradio interface
iface = gr.Interface(
    fn=stock_analysis,
    inputs=[
        gr.Number(label="DBS Stock Price", value=38, precision=2),
        gr.Dropdown(
            label="Select Stock to Compare with DBS",
            choices=["DBS", "UOB", "OCBC", "Singtel", "Standard Chartered", "HSBC"],
            value="UOB"  # Default comparison stock
        )
    ],
    outputs=[
        gr.Textbox(label="Predicted Stock Price"),
        gr.Textbox(label="Linear Regression Equation (y = mx + c)"),
        gr.Image(label="Linear Regression Plot")
    ],
    live=True,  # Ensures live update
    allow_flagging="never"  # Disable the flag button as it's not useful here
)

# Launch the Gradio app
iface.launch()
