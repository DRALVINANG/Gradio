# Install necessary libraries
!pip install gradio seaborn matplotlib scikit-learn pandas

# Import necessary libraries
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the automobile dataset from the provided URL
url = 'https://www.alvinang.sg/s/automobileEDA.csv'
df = pd.read_csv(url)

# Define X and Y
X = df[['highway-mpg']]
y = df['price']

# Initialize the Linear Regression model
lm = LinearRegression()
lm.fit(X, y)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
def predict_price(highway_mpg):
    """
    Predict the car price based on highway-mpg using the trained linear regression model.
    """
    # Predict the price for the given highway-mpg
    predicted_price = lm.predict([[highway_mpg]])[0]

    # Create a regression plot showing highway-mpg vs. price
    plt.figure(figsize=(12, 8))
    sns.regplot(x='highway-mpg', y='price', data=df, line_kws={'color': 'red'})
    plt.scatter([highway_mpg], [predicted_price], color='orange', label=f'Predicted Price: ${predicted_price:.2f}', s=100)
    plt.xlabel('Highway-MPG')
    plt.ylabel('Price')
    plt.title('Regression Plot: Highway-MPG vs Price')
    plt.legend()
    plt.grid()

    # Save the plot to a file for Gradio to display
    plot_path = "regression_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Return predicted price, R-squared, MSE, and the regression plot
    r2 = lm.score(X, y)
    mse = mean_squared_error(y, lm.predict(X))
    return f"${predicted_price:.2f}", f"{r2:.2f}", f"{mse:.2f}", plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Automobile Price Prediction Based on Highway-MPG")
    gr.Markdown("""
    This app uses a linear regression model to predict the price of automobiles based on their highway-mpg value.
    You can adjust the highway-mpg slider below to predict the price and visualize the regression line.
    """)

    with gr.Row():
        highway_mpg_input = gr.Slider(label="Highway-MPG", minimum=int(X.min()), maximum=int(X.max()), step=1, value=30)
    
    with gr.Row():
        predicted_price_output = gr.Textbox(label="Predicted Price (USD)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)

    plot_output = gr.Image(label="Regression Plot")

    predict_button = gr.Button("Predict and Visualize")

    # Connect function to Gradio components
    predict_button.click(
        predict_price,
        inputs=[highway_mpg_input],
        outputs=[predicted_price_output, r2_output, mse_output, plot_output]
    )

# Launch the app with share=True for Colab
demo.launch(share=True)
