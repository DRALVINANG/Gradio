# Install required libraries
!pip install gradio
!pip install yfinance
!pip install scikit-learn

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import gradio as gr

# Function to download stock data and perform linear regression
def stock_analysis(dbs_price):
    start_date = '2020-01-01'

    # Download stock data for DBS and UOB
    data1 = yf.download('D05.SI', start=start_date)  # DBS
    data2 = yf.download('U11.SI', start=start_date)  # UOB

    data1.columns = data1.columns.droplevel(level=1)
    data2.columns = data2.columns.droplevel(level=1)

    # Merge both stock prices into one dataframe
    data = pd.merge(data1.Close, data2.Close, left_index=True, right_index=True)
    data.rename(columns={'Close_x': 'DBS Close Price', 'Close_y': 'UOB Close Price'}, inplace=True)

    # Linear Regression Model
    y = data['UOB Close Price'].values.reshape(-1, 1)
    X = data['DBS Close Price'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    # Predict UOB price based on DBS price
    predicted_value = model.predict([[dbs_price]])

    # Plot the scatter plot and regression line
    plt.figure(figsize=(12, 5))
    plt.scatter(data['DBS Close Price'], data['UOB Close Price'], label='Data Points')
    plt.plot(data['DBS Close Price'], model.intercept_[0] + model.coef_[0][0] * data['DBS Close Price'], color='red', label='Regression Line')
    plt.xlabel('DBS Price')
    plt.ylabel('UOB Price')
    plt.title('Linear Regression: DBS vs UOB Stock Prices')
    plt.legend()

    # Save the plot to a file
    plot_filename = "/tmp/stock_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    # Return the prediction and plot filename
    return f"Predicted UOB Close Price for DBS price of {dbs_price}: {predicted_value[0][0]:.2f}", plot_filename

# Create the Gradio interface
iface = gr.Interface(
    fn=stock_analysis,
    inputs=[
        gr.Number(label="DBS Stock Price", value=38, precision=2)
    ],
    outputs=[
        gr.Textbox(label="Predicted UOB Price"),
        gr.Image(label="Linear Regression Plot")
    ],
    live=True,  # Ensures live update
    allow_flagging="never"  # Disable the flag button as it's not useful here
)

# Launch the Gradio app
iface.launch()
