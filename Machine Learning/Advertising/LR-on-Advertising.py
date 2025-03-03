import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy DataFrame for dataset preview (replace with actual data)
data = {
    "TV Advertising Budget ($)": [100, 150, 200, 250, 300],
    "Sales (units)": [20, 25, 30, 35, 40]
}
advert = pd.DataFrame(data)

# Train a linear regression model
X = advert[["TV Advertising Budget ($)"]]
y = advert["Sales (units)"]
model = LinearRegression()
model.fit(X, y)
intercept = model.intercept_
slope = model.coef_[0]

# Function to generate visualizations
def generate_visualizations():
    # Scatter Plot
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

# Function to predict sales based on TV advertising budget
def predict_and_visualize_with_feedback(tv_budget):
    # Predict sales using trained model
    predicted_sales = model.predict(np.array([[tv_budget]]))[0]
    r2_value = model.score(X, y)
    mse_value = np.mean((model.predict(X) - y) ** 2)

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
    x_vals = np.linspace(0, 500, 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red", label="Regression Line")
    plt.title("Regression Plot")
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
    gr.Markdown("# TV Advertising vs Sales - Linear Regression")
    gr.Markdown("""
    **Objective:** This app predicts sales based on TV advertising budgets using linear regression.
    """)

    # Dataset Preview
    gr.Markdown("### Dataset Preview:")
    dataset_preview = gr.Dataframe(value=advert.head())
    
    # Visualization Outputs
    pair_plot_output = gr.Image()
    heatmap_output = gr.Image()
    generate_visualizations_button = gr.Button("Generate Visualizations")
    generate_visualizations_button.click(
        generate_visualizations,
        inputs=[],
        outputs=[pair_plot_output, heatmap_output]
    )

    # Input for prediction
    tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
    predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
    r2_output = gr.Textbox(label="R-squared Value", interactive=False)
    mse_output = gr.Textbox(label="Mean Squared Error", interactive=False)
    plot_output = gr.Image()

    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_and_visualize_with_feedback,
        inputs=tv_budget_input,
        outputs=[predicted_sales_output, r2_output, mse_output, plot_output]
    )

# Launch app
demo.launch(share=True)
