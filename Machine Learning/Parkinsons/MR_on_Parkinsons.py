# Install necessary libraries
!pip install gradio seaborn matplotlib scikit-learn pandas

# Import necessary libraries
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Parkinson's dataset from the provided URL
dataset_path = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Multiple%20Regression/Parkinsons.csv"
data = pd.read_csv(dataset_path)

# Drop unnecessary columns
data = data.dropna()  # Drop missing values
data = data.drop(columns=['subject#'])  # Remove subject ID as it's irrelevant to prediction

# Separate features (X) and target (y)
X = data.drop(columns=['total_UPDRS', 'motor_UPDRS'])  # Features
y = data['total_UPDRS']  # Target variable

# Scale features to a range between 0 and 1
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate R² and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
# Function to generate and save pair plot and correlation heatmap
def generate_visualizations():
    # Generate pair plot for selected features
    subset = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'total_UPDRS']]
    sns.pairplot(subset)
    pair_plot_path = "pair_plot.png"
    plt.savefig(pair_plot_path)
    plt.close()

    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    return pair_plot_path, heatmap_path

# Function to make predictions and visualize results
def predict_and_visualize(jitter, shimmer, nhr, hnr, rpde, dfa, ppe):
    # Create a single input row with scaled features
    input_data = pd.DataFrame({
        "Jitter(%)": [jitter],
        "Shimmer": [shimmer],
        "NHR": [nhr],
        "HNR": [hnr],
        "RPDE": [rpde],
        "DFA": [dfa],
        "PPE": [ppe],
    })

    # Scale the input data
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # Predict the total UPDRS
    predicted_total_UPDRS = model.predict(input_data_scaled)[0]

    # Create a residual plot
    plt.figure(figsize=(12, 6))
    sns.residplot(x=y_test, y=y_pred, lowess=True, color="g")
    plt.xlabel('Actual UPDRS')
    plt.ylabel('Residuals')
    plt.title('Residual Plot: Actual vs Predicted UPDRS')
    residual_plot_path = "residual_plot.png"
    plt.savefig(residual_plot_path)
    plt.close()

    # Explain model performance
    mse_comment = "Good Fit" if mse <= 10 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"

    return round(predicted_total_UPDRS, 2), f"{r2:.2f} ({r2_comment})", f"{mse:.2f} ({mse_comment})", residual_plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
# Create a Gradio app interface
with gr.Blocks() as demo:
    gr.Markdown("# Parkinson's Disease Prediction Using Linear Regression")
    gr.Markdown("""
    ## About the Dataset:
    The **Parkinson’s Telemonitoring Dataset** provides biomedical voice measurements to monitor the progression of Parkinson's disease. 
    The dataset includes features related to vocal variability, noise, and entropy, and is used for predicting the severity of the disease.

    **Target Variable**:
    - **Total UPDRS**: Reflects overall severity of Parkinson's disease (target variable for this model).

    **Dataset Summary**:
    - **Rows**: 5,875
    - **Features**: 22
    - **Target**: Total UPDRS
    """)

    gr.Markdown("""
    ## Visualize Relationships:
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows the relationship between selected features and Total UPDRS.
    - **Correlation Heatmap**: Displays correlations among all features.
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
    1. Adjust the feature values below using sliders.
    2. The app will predict the **Total UPDRS** and display performance metrics:
        - **R² Score**: Indicates model fit (higher is better).
        - **Mean Squared Error (MSE)**: Measures prediction error (lower is better).

    ### Performance Guidelines:
    - **R² Score**:
        - > 0.9: Excellent Fit
        - 0.7 - 0.9: Acceptable Fit
        - ≤ 0.7: Poor Fit
    - **MSE**:
        - ≤ 10: Good Fit
        - > 100: Poor Fit
    """)

    # Define sliders for key input features
    sliders = [
        gr.Slider(0, 1, step=0.01, label="Jitter (%)", value=0.01),
        gr.Slider(0, 1, step=0.01, label="Shimmer", value=0.05),
        gr.Slider(0, 1, step=0.01, label="NHR", value=0.1),
        gr.Slider(0, 50, step=1, label="HNR", value=20),
        gr.Slider(0, 1, step=0.01, label="RPDE", value=0.5),
        gr.Slider(0, 1, step=0.01, label="DFA", value=0.6),
        gr.Slider(0, 1, step=0.01, label="PPE", value=0.2),
    ]

    with gr.Row():
        predicted_updrs = gr.Textbox(label="Predicted Total UPDRS")
        r2_output = gr.Textbox(label="R² Score")
        mse_output = gr.Textbox(label="MSE")
    residual_plot_output = gr.Image(label="Residual Plot")

    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_and_visualize,
        inputs=sliders,
        outputs=[predicted_updrs, r2_output, mse_output, residual_plot_output]
    )

# Launch the app with share=True for Colab
demo.launch(share=True)
