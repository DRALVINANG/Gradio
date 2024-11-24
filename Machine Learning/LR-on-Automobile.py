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

# Predict values and calculate R² and MSE
r2 = lm.score(X, y)
mse = mean_squared_error(y, lm.predict(X))

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
# Function to generate and save pair plot and correlation heatmap
def generate_visualizations():
    # Generate pair plot
    sns.pairplot(df[['highway-mpg', 'price']])
    pair_plot_path = "pair_plot.png"
    plt.savefig(pair_plot_path)
    plt.close()

    # Generate correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[['highway-mpg', 'price']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    return pair_plot_path, heatmap_path

# Function to make predictions and visualize results
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

    # Explain model performance
    mse_comment = "Good Fit" if mse <= 10 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"

    return f"${predicted_price:.2f}", f"{r2:.2f} ({r2_comment})", f"{mse:.2f} ({mse_comment})", plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Automobile Price Prediction Based on Highway-MPG")
    gr.Markdown("""
    ## About the Dataset:
    The **Automobile Dataset** provides data for predicting the price of different types of cars based on their various attributes. 
    The dataset contains 26 features (attributes), but here, we focus on predicting price using **highway-mpg**.

    - **Highway MPG (highway-mpg)**: Fuel efficiency of the car in highway driving conditions.
    - **Price**: The price of the car in dollars (target variable).

    **Dataset Summary**:
    - **Rows**: 205 (cars)
    - **Columns**: 2 (highway-mpg, price)
    """)

    gr.Markdown("""
    ## Visualize Relationships:
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows the relationship between highway-mpg and price.
    - **Correlation Heatmap**: Displays the correlation coefficient between highway-mpg and price.
    """)

    pair_plot_output = gr.Image(label="Pair Plot")
    heatmap_output = gr.Image(label="Correlation Heatmap")
    generate_visualizations_button = gr.Button("Generate Visualizations")
    generate_visualizations_button.click(
        generate_visualizations, 
        inputs=[], 
        outputs=[pair_plot_output, heatmap_output]
    )

    gr.Markdown("""
    ## How to Use This App:
    1. Adjust the **highway-mpg** slider below to predict the car price.
    2. The app will display:
        - The **predicted price** of the car.
        - **R² Score (R-squared)**: Indicates how well the model fits the data. Higher values (closer to 1) are better.
        - **Mean Squared Error (MSE)**: Indicates the average squared difference between actual and predicted values. Lower values are better.

    ### Performance Guidelines:
    - **R² Score:**
        - > 0.9: Excellent Fit
        - 0.7 - 0.9: Acceptable Fit
        - ≤ 0.7: Poor Fit
    - **MSE:**
        - ≤ 10: Good Fit
        - > 100: Poor Fit
    """)

    with gr.Row():
        highway_mpg_input = gr.Slider(label="Highway-MPG", minimum=int(X.min()), maximum=int(X.max()), step=1, value=30)
    
    with gr.Row():
        predicted_price_output = gr.Textbox(label="Predicted Price (USD)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)

    plot_output = gr.Image(label="Regression Plot")

    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_price,
        inputs=[highway_mpg_input],
        outputs=[predicted_price_output, r2_output, mse_output, plot_output]
    )

# Launch the app with share=True for Colab
demo.launch(share=True)
