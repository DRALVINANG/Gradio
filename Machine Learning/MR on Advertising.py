# Install necessary libraries
!pip install gradio pandas matplotlib seaborn scikit-learn

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the Advertising dataset from the provided URL
url = 'https://www.alvinang.sg/s/Advertising.csv'
advert = pd.read_csv(url)

# Function to generate visualizations
def generate_visualizations():
    # Pair Plot
    plt.figure(figsize=(5, 5))
    sns.pairplot(advert)
    plt.savefig("pair_plot.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(advert.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("heatmap.png")
    plt.close()

    return "pair_plot.png", "heatmap.png"

# Function to predict and visualize results with feedback
def predict_and_visualize_with_feedback(tv_budget, radio_budget, newspaper_budget):
    # Prepare data for prediction
    X = advert[['TV', 'Radio', 'Newspaper']]
    y = advert['Sales']

    # Split the dataset into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict sales based on user input
    new_X = [[tv_budget, radio_budget, newspaper_budget]]
    predicted_sales = model.predict(new_X)[0]

    # Predict on test set for evaluation metrics
    sales_pred_test = model.predict(X_test)

    # Calculate R-squared value and MSE for feedback
    r2_value = r2_score(y_test, sales_pred_test)
    mse_value = ((y_test - sales_pred_test) ** 2).mean()

    # Feedback for R²
    if r2_value > 0.9:
        r2_feedback = "Excellent Fit"
    elif 0.7 <= r2_value <= 0.9:
        r2_feedback = "Acceptable Fit"
    else:
        r2_feedback = "Poor Fit"

    # Feedback for MSE
    if mse_value <= 10:
        mse_feedback = "Good Fit"
    elif mse_value > 100:
        mse_feedback = "Poor Fit"
    else:
        mse_feedback = "Moderate Fit"

    # Regression plot
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=sales_pred_test)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", label="Ideal Prediction")
    plt.title("Regression Plot")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.legend()
    plt.savefig("regression_plot.png")
    plt.close()

    return (
        predicted_sales,
        f"{r2_value:.2f} ({r2_feedback})",
        f"{mse_value:.2f} ({mse_feedback})",
        "regression_plot.png"
    )

# Gradio Interface
with gr.Blocks() as demo:
    # Main Title
    gr.Markdown("# Advertising Budget vs Sales - Multiple Linear Regression")

    gr.Markdown("""
    **Objective:** This app demonstrates how to use Multiple Linear Regression to predict sales based on TV, Radio, and Newspaper advertising budgets. It includes interactive visualizations and performance metrics to guide users through the model's behavior and insights.

    **Created by:** Dr. Alvin Ang
    """)

    gr.Markdown("<hr>")  # Add a horizontal line after the title and description

    # About the Dataset Section
    gr.Markdown("## About the Dataset:")

    gr.Markdown("""
    This dataset consists of sales data for a product across different markets. It includes **TV**, **Radio**, and **Newspaper** advertising budgets along with corresponding **sales** for each market.

    - **TV**: Advertising budget spent on TV (in $).
    - **Radio**: Advertising budget spent on Radio (in $).
    - **Newspaper**: Advertising budget spent on Newspaper (in $).
    - **Sales**: The number of units sold in each market (target variable).
    """)

    # Dataset Preview Section
    gr.Markdown("### Dataset Preview:")
    dataset_preview = gr.Dataframe(label="Dataset Preview", value=advert.head())

    # Add a hyperlink to download the dataset
    gr.Markdown("""
    [Download the Dataset](https://www.alvinang.sg/s/Advertising.csv)
    """)

    gr.Markdown("<hr>")  # Add a horizontal line after the dataset preview and download link

    # Visualization Section
    gr.Markdown("## Visualize Relationships:")
    gr.Markdown("""
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows the relationship between advertising budgets and sales.
    - **Correlation Heatmap**: Displays the correlation coefficient between features.
    """)

    # Visualization Outputs
    pair_plot_output = gr.Image(label="Pair Plot")
    heatmap_output = gr.Image(label="Correlation Heatmap")
    generate_visualizations_button = gr.Button("Generate Visualizations")
    generate_visualizations_button.click(
        generate_visualizations,
        inputs=[],
        outputs=[pair_plot_output, heatmap_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after visualizations

    # Instructions Section
    gr.Markdown("## How to Use This App:")
    gr.Markdown("""
    1. Adjust the **advertising budgets** using the sliders below.
    2. The app will predict the **sales** and display performance metrics:
        - **R² Score (R-squared)**: Measures how well the model fits the data. Higher values (closer to 1) indicate a better fit.
        - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values. Lower values (close to 0) indicate a better fit.

    ### Performance Guidelines:
    - **R² Score:**
        - > 0.9: Excellent Fit
        - 0.7 - 0.9: Acceptable Fit
        - ≤ 0.7: Poor Fit
    - **MSE:**
        - ≤ 10: Good Fit
        - > 100: Poor Fit
    """)

    gr.Markdown("<hr>")  # Add a horizontal line after instructions

    # Input and Outputs Section for Prediction
    gr.Markdown("### Make Predictions:")

    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
        radio_budget_input = gr.Slider(label="Radio Advertising Budget ($)", minimum=0, maximum=500, step=10, value=50)
        newspaper_budget_input = gr.Slider(label="Newspaper Advertising Budget ($)", minimum=0, maximum=500, step=10, value=20)

    gr.Markdown("### Predicted Results and Model Performance:")

    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value and Feedback", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE) and Feedback", interactive=False)

    plot_output = gr.Image(label="Regression Plot")

    # Button to make predictions and visualize results
    predict_button = gr.Button("Predict and Visualize")

    predict_button.click(
        predict_and_visualize_with_feedback,
        inputs=[tv_budget_input, radio_budget_input, newspaper_budget_input],
        outputs=[predicted_sales_output, r2_output, mse_output, plot_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after predictions and performance section

# Launch the app with share=True for Colab or local testing
demo.launch(share=True)
