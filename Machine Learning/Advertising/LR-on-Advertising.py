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

    # Add the uploaded image below the title
    gr.Image(value="/mnt/data/1.png", label="TV Advertising")

    gr.Markdown("<hr>")  # Add a horizontal line after the title and description

    # About the Dataset Section
    gr.Markdown("## About the Dataset:")
    gr.Markdown("""
    This dataset consists of sales data for a product across 200 different markets. It includes the **TV advertising budget** and the corresponding **sales** for each market.

    - **TV Advertising Budget (TV)**: Advertising budget spent on TV (in $).
    - **Sales**: The number of units sold in each market (target variable).

    **Dataset Summary**:
    - **200 rows** (markets) and **2 variables**:
        - TV: Advertising budget for TV (numeric, feature).
        - Sales: Product sales (numeric, target variable).
    """)
    gr.Markdown("<hr>")  # Add a horizontal line after the dataset description

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
    gr.Markdown("<hr>")  # Add a horizontal line after the instructions

    # Input and Outputs Section for Prediction
    gr.Markdown("### Make Predictions:")
    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)

    gr.Markdown("### Predicted Results and Model Performance:")
    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)

    plot_output = gr.Image(label="Regression Plot")

    # Button to make predictions and visualize results
    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(predict_and_visualize, inputs=tv_budget_input, outputs=[predicted_sales_output, r2_output, mse_output, plot_output])
    gr.Markdown("<hr>")  # Add a horizontal line after predictions and performance section

# Launch the app with share=True for Colab
demo.launch(share=True)
