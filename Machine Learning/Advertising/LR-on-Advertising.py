# Install required libraries
!pip install gradio pandas matplotlib seaborn

# Import necessary libraries
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy DataFrame for dataset preview (replace with your actual data)
data = {
    "TV Advertising Budget ($)": [100, 150, 200, 250, 300],
    "Sales (units)": [20, 25, 30, 35, 40]
}
advert = pd.DataFrame(data)

# Function to generate visualizations
def generate_visualizations():
    # Pair Plot
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=advert, x="TV Advertising Budget ($)", y="Sales (units)")
    plt.title("Scatter Plot of TV Advertising vs Sales")
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
def predict_and_visualize_with_feedback(tv_budget):
    # Dummy linear regression prediction logic
    predicted_sales = tv_budget * 0.2  # Dummy prediction formula
    r2_value = 0.95  # Dummy R² value
    mse_value = 5.0  # Dummy MSE value

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
    sns.scatterplot(data=advert, x="TV Advertising Budget ($)", y="Sales (units)")
    plt.plot([0, 500], [0, 500 * 0.2], color="red", label="Regression Line")
    plt.title("Regression Plot")
    plt.legend()
    plt.savefig("regression_plot.png")
    plt.close()

    # Return values and feedback
    return (
        predicted_sales,
        f"{r2_value:.2f} ({r2_feedback})",
        f"{mse_value:.2f} ({mse_feedback})",
        "regression_plot.png"
    )

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    # Main Title
    gr.Markdown("# TV Advertising vs Sales - Linear Regression")
    gr.Markdown("""
    **Objective:** This app demonstrates how to use Linear Regression to predict sales based on TV advertising budgets. It includes interactive visualizations, predictions, and performance metrics to guide users through the model's behavior and insights.

    **Created by:** Dr. Alvin Ang
    """)

    gr.Markdown("<hr>")  # Add a horizontal line after the title and description

    # About the Dataset Section
    gr.Markdown("## About the Dataset:")
    gr.Markdown("""
    This dataset consists of sales data for a product across 200 different markets. It includes the **TV advertising budget** and the corresponding **sales** for each market.

    - **TV Advertising Budget (TV)**: Advertising budget spent on TV (in $).
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
    - **Pair Plot**: Shows the relationship between TV advertising budgets and sales.
    - **Correlation Heatmap**: Displays the correlation coefficient between TV and sales.
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
    1. Adjust the **TV advertising budget** using the slider below.
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
        inputs=tv_budget_input,
        outputs=[predicted_sales_output, r2_output, mse_output, plot_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after predictions and performance section

# Launch the app with share=True for Colab
demo.launch(share=True)
