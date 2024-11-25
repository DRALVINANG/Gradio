# Install required libraries
!pip install gradio pandas matplotlib seaborn scikit-learn

# Import necessary libraries
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import requests

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Advertising dataset
url = 'https://www.alvinang.sg/s/Advertising.csv'
advert = pd.read_csv(url)

# Define X (features) and y (target)
X = advert[['TV', 'Radio', 'Newspaper']]
y = advert['Sales']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values for the test set
sales_pred_test = model.predict(X_test)

# Calculate the R-squared value and Mean Squared Error
r2 = r2_score(y_test, sales_pred_test)
mse = mean_squared_error(y_test, sales_pred_test)

# Save the image locally
image_url = "https://raw.githubusercontent.com/DRALVINANG/Gradio/main/Machine%20Learning/Advertising/pexels-jvdm-1457842.jpg"
image_path = "advertising_image.jpg"
with open(image_path, "wb") as f:
    f.write(requests.get(image_url).content)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
# Function to display the image
def load_image():
    return image_path

# Function to generate visualizations
def generate_visualizations():
    # Generate pair plot
    sns.pairplot(advert)
    pair_plot_path = "pair_plot.png"
    plt.savefig(pair_plot_path)
    plt.close()

    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(advert.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    return pair_plot_path, heatmap_path

# Function to predict and visualize results
def predict_and_visualize(tv_budget, radio_budget, newspaper_budget):
    # Predict sales for the input budgets
    predicted_sales = model.predict([[tv_budget, radio_budget, newspaper_budget]])[0]

    # R² and MSE feedback
    r2_feedback = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"
    mse_feedback = "Good Fit" if mse <= 10 else "Poor Fit"

    return predicted_sales, f"{r2:.2f} ({r2_feedback})", f"{mse:.2f} ({mse_feedback})"

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
# Create a Gradio app interface
with gr.Blocks() as demo:
    # Main Title
    gr.Markdown("# Advertising Campaigns vs Sales - Multiple Linear Regression")
    gr.Markdown("""
    **Objective:** This app demonstrates how advertising budgets for TV, Radio, and Newspapers influence sales. It uses a Multiple Linear Regression model to predict sales and provides insights through visualizations.
    """)

    # Add a button to display the image
    gr.Markdown("### Click below to display an example advertising image:")
    load_image_button = gr.Button("Show Image")
    advertising_image_output = gr.Image(label="Example Advertising Image")
    load_image_button.click(load_image, inputs=[], outputs=[advertising_image_output])

    gr.Markdown("<hr>")  # Horizontal line after the image

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
    
    # Add a dataset preview
    gr.Markdown("### Dataset Preview:")
    dataset_preview = gr.Dataframe(value=advert.head(), label="First Few Rows of Dataset")

    # Add a download button for the dataset
    def download_csv():
        return url  # Provide the CSV download link

    download_button = gr.Button("Download Dataset")
    dataset_download_output = gr.File(label="Download CSV")
    download_button.click(download_csv, inputs=[], outputs=dataset_download_output)

    gr.Markdown("<hr>")  # Horizontal line after the dataset preview and download button

    gr.Markdown("## Visualize Relationships:")
    gr.Markdown("""
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows relationships between all variables.
    - **Correlation Heatmap**: Displays correlation coefficients between variables.
    """)
    
    pair_plot_output = gr.Image(label="Pair Plot")
    heatmap_output = gr.Image(label="Correlation Heatmap")
    generate_visualizations_button = gr.Button("Generate Visualizations")
    generate_visualizations_button.click(
        generate_visualizations, 
        inputs=[], 
        outputs=[pair_plot_output, heatmap_output]
    )

    gr.Markdown("<hr>")  # Horizontal line after visualizations

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
