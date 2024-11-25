# Install required libraries
!pip install gradio pandas matplotlib seaborn

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
# Load the Parkinson's dataset
dataset_path = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Multiple%20Regression/Parkinsons.csv"
data = pd.read_csv(dataset_path)

# Save dataset locally for download
dataset_local_path = "parkinsons_dataset.csv"
data.to_csv(dataset_local_path, index=False)

# Process dataset
data = data.dropna()  # Drop missing values
data = data.drop(columns=['subject#'])  # Remove subject ID column
X = data.drop(columns=['total_UPDRS', 'motor_UPDRS'])
y = data['total_UPDRS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for the test set
y_pred = model.predict(X_test)

# Calculate R² and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save the image locally
image_url = "https://github.com/DRALVINANG/Gradio/blob/main/Machine-Learning/Parkinsons/pexels-chokniti-khongchum-1197604-3938022.jpg?raw=true"
image_path = "parkinsons_disease.jpg"
response = requests.get(image_url)
with open(image_path, "wb") as f:
    f.write(response.content)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
# Function to display the image
def load_image():
    return image_path

# Function to generate visualizations
def generate_visualizations():
    # Pair Plot
    subset = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'total_UPDRS']]
    sns.pairplot(subset)
    plt.savefig("pair_plot.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

    return "pair_plot.png", "correlation_heatmap.png"

# Function to make predictions and generate residual plot
def predict_and_visualize(jitter, shimmer, nhr, hnr, rpde, dfa, ppe):
    # Predict total UPDRS
    user_input = [[jitter, shimmer, nhr, hnr, rpde, dfa, ppe]]
    predicted_total_UPDRS = model.predict(user_input)[0]

    # Residual Calculation
    residuals = y_test - y_pred  # Actual - Predicted

    # Residual Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=residuals)  # Residual plot
    plt.axhline(0, color="red", linestyle="--", linewidth=1)  # Add horizontal line at residual=0
    plt.title("Residual Plot")
    plt.xlabel("Actual Total UPDRS")
    plt.ylabel("Residuals")
    plt.savefig("residual_plot.png")
    plt.close()

    # Feedback
    r2_feedback = "Acceptable Fit" if 0.7 <= r2 <= 0.9 else "Poor Fit"
    mse_feedback = "Good Fit" if mse <= 10 else "Moderate Fit"

    return round(predicted_total_UPDRS, 2), f"{r2:.2f} ({r2_feedback})", f"{mse:.2f} ({mse_feedback})", "residual_plot.png"

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    # Main Title
    gr.Markdown("# Parkinson's Disease Prediction Using Linear Regression")
    gr.Markdown("""
    This app predicts the progression severity of Parkinson's disease using linear regression models. It provides interactive visualizations, predictions, and model performance metrics to help users understand the data and model behavior.
    """)
    gr.Markdown("<hr>")  # Add a horizontal line

    # Add a button to load an image
    gr.Markdown("### Click the button below to load an image related to Parkinson's Disease:")
    load_image_button = gr.Button("Load Image")
    disease_image_output = gr.Image(label="Parkinson's Disease")
    load_image_button.click(load_image, inputs=[], outputs=disease_image_output)

    gr.Markdown("<hr>")  # Add a horizontal line

    # About the Dataset
    gr.Markdown("## About the Dataset:")
    gr.Markdown("""
    The **Parkinson’s Telemonitoring Dataset** from the UCI Machine Learning Repository provides data for monitoring the progression of Parkinson’s disease based on various biomedical voice measurements.
    
    **Features:**
    - **Jitter(%):** Variation in frequency measured as a percentage.
    - **Shimmer:** Variation in amplitude, showing amplitude differences.
    - **NHR (Noise-to-Harmonics Ratio):** Ratio indicating the noise level relative to harmonic energy.
    - **HNR (Harmonics-to-Noise Ratio):** A measure of vocal clarity.
    - **RPDE (Recurrence Period Density Entropy):** Nonlinear dynamic feature measuring vocal signal unpredictability.
    - **DFA (Detrended Fluctuation Analysis):** A measure of signal self-similarity.
    - **PPE (Pitch Period Entropy):** Entropy of pitch periods, representing vocal variability.

    **Target Variables:**
    - **Motor UPDRS:** Reflects motor symptoms severity.
    - **Total UPDRS:** Overall severity of Parkinson’s disease, including motor and non-motor symptoms.
    """)

    # Dataset Preview
    gr.Markdown("### Dataset Preview:")
    dataset_preview_table = gr.Dataframe(value=data.head(), label="Dataset Preview")
    download_button = gr.File(label="Download Parkinson's Dataset", value=dataset_local_path)

    gr.Markdown("<hr>")  # Add a horizontal line

    # Visualization Section
    gr.Markdown("## Visualize Relationships:")
    pair_plot_output = gr.Image(label="Pair Plot")
    heatmap_output = gr.Image(label="Correlation Heatmap")
    generate_visualizations_button = gr.Button("Generate Visualizations")
    generate_visualizations_button.click(
        generate_visualizations,
        inputs=[],
        outputs=[pair_plot_output, heatmap_output]
    )
    gr.Markdown("<hr>")  # Add a horizontal line

    # Prediction Section
    gr.Markdown("### Predict Total UPDRS:")
    sliders = [
        gr.Slider(0, 1, step=0.01, label="Jitter (%)", value=0.01),
        gr.Slider(0, 1, step=0.01, label="Shimmer", value=0.05),
        gr.Slider(0, 1, step=0.01, label="NHR", value=0.1),
        gr.Slider(0, 50, step=1, label="HNR", value=20),
        gr.Slider(0, 1, step=0.01, label="RPDE", value=0.5),
        gr.Slider(0, 1, step=0.01, label="DFA", value=0.6),
        gr.Slider(0, 1, step=0.01, label="PPE", value=0.2),
    ]
    predicted_updrs = gr.Textbox(label="Predicted Total UPDRS")
    r2_output = gr.Textbox(label="R² Score and Feedback")
    mse_output = gr.Textbox(label="Mean Squared Error and Feedback")
    residual_plot_output = gr.Image(label="Residual Plot")
    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_and_visualize,
        inputs=sliders,
        outputs=[predicted_updrs, r2_output, mse_output, residual_plot_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after predictions

# Launch the app with share=True
demo.launch(share=True)
