#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    # Main Title
    gr.Markdown("# Advertising Campaigns vs Sales - Multiple Linear Regression")
    gr.Markdown("""**Objective:** This app demonstrates how advertising budgets for TV, Radio, and Newspapers influence sales. It uses a Multiple Linear Regression model to predict sales and provides insights through visualizations.""")

    gr.Markdown("<hr>")  # Horizontal line after the title and description

    # About the Dataset Section
    gr.Markdown("## About the Dataset:")
    gr.Markdown("""
    This dataset provides data on advertising budgets for TV, Radio, and Newspapers, along with the resulting sales. The goal is to analyze and predict how different budget allocations impact sales.

    - **Features:**
        - TV: Budget for TV advertising (in $).
        - Radio: Budget for radio advertising (in $).
        - Newspaper: Budget for newspaper advertising (in $).
    - **Target:**
        - Sales: Product sales in units.
    """)

    # Dataset Preview Section
    gr.Markdown("### Dataset Preview:")
    dataset_preview = gr.Dataframe(value=advert.head(), label="First Few Rows of Dataset")

    # Add a hyperlink to download the dataset
    gr.Markdown("""
    [Download the Full Dataset](https://www.alvinang.sg/s/Advertising.csv)
    """)

    gr.Markdown("<hr>")  # Horizontal line after the dataset preview and download link

    # Visualization Section
    gr.Markdown("## Visualize Relationships:")
    gr.Markdown("""
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows relationships between all variables.
    - **Correlation Heatmap**: Displays correlation coefficients between variables.
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

    gr.Markdown("<hr>")  # Horizontal line after visualizations

    # Make Predictions Section
    gr.Markdown("## Make Predictions:")
    gr.Markdown("""
    Adjust the advertising budgets using the sliders below, and the app will predict the sales and display performance metrics:
    - **R² Score (R-squared):** Measures how well the model fits the data. Higher values indicate a better fit.
    - **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Lower values indicate a better fit.
    """)

    # Input sliders for budgets
    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
        radio_budget_input = gr.Slider(label="Radio Advertising Budget ($)", minimum=0, maximum=500, step=10, value=50)
        newspaper_budget_input = gr.Slider(label="Newspaper Advertising Budget ($)", minimum=0, maximum=500, step=10, value=20)
    
    # Outputs for predictions and metrics
    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)

    # Button to make predictions
    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_and_visualize,
        inputs=[tv_budget_input, radio_budget_input, newspaper_budget_input],
        outputs=[predicted_sales_output, r2_output, mse_output]
    )

    gr.Markdown("<hr>")  # Horizontal line after predictions

# Launch the app with share=True
demo.launch(share=True)
