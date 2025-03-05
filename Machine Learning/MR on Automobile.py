# Install necessary libraries
!pip install gradio pandas matplotlib seaborn scikit-learn

# Import necessary libraries
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Automobile dataset
path = 'https://www.alvinang.sg/s/automobileEDA.csv'
df = pd.read_csv(path)

# Define features (X) and target (y)
X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = df['price']

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------

# Function to generate pair plot and correlation heatmap
def generate_visualizations():
    # Pair Plot
    plt.figure(figsize=(10, 8))
    sns.pairplot(df)
    plt.savefig("pair_plot.png")
    plt.close()

    # Filter numeric columns for the correlation heatmap
    numeric_df = df.select_dtypes(include=[float, int])

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("heatmap.png")
    plt.close()

    return "pair_plot.png", "heatmap.png"

# Function to make predictions and visualize results
def predict_and_visualize(horsepower, curb_weight, engine_size, highway_mpg):
    # Predict the price
    predicted_price = model.predict([[horsepower, curb_weight, engine_size, highway_mpg]])[0]

    # Predict for the whole dataset for visualization
    y_pred = model.predict(X)

    # Create Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['price'], color="r", label="Actual Price")
    sns.kdeplot(y_pred, color="b", label="Predicted Price")
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("actual_vs_predicted.png")
    plt.close()

    # Create Residual Plot
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig("residual_plot.png")
    plt.close()

    # Calculate evaluation metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    mse_comment = "Good Fit" if mse <= 1e6 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"

    return predicted_price, f"{r2:.2f} ({r2_comment})", f"{mse:.2f} ({mse_comment})", "actual_vs_predicted.png", "residual_plot.png"

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Automobile Price Prediction - Linear Regression")

    gr.Markdown("""
        **Objective:** This app uses **Linear Regression** to predict the price of a car based on its attributes.
        The model is trained using features like **horsepower**, **curb weight**, **engine size**, and **highway-mpg**.
        By adjusting these values, users can predict the price of an automobile and visualize how the model performs.
        """)

    gr.Markdown("<hr>")  # Add a horizontal line after the title and description

    # Dataset Preview Section
    gr.Markdown("## Dataset Preview:")

    dataset_preview = gr.Dataframe(value=df.head(), interactive=False)

    gr.Markdown("<hr>")  # Add a horizontal line after the "Dataset Preview" section

    # Download Dataset Button
    gr.Markdown("### Download the Dataset:")

    gr.Markdown("[Click here to download the dataset](https://www.alvinang.sg/s/automobileEDA.csv)")

    gr.Markdown("<hr>")  # Add a horizontal line after the "Download Dataset" section

    # Visualize Relationships
    gr.Markdown("## Visualize Relationships")

    pair_plot_output = gr.Image(label="Pair Plot")
    heatmap_output = gr.Image(label="Correlation Heatmap")

    generate_visualizations_button = gr.Button("Generate Visualizations")

    generate_visualizations_button.click(
        generate_visualizations,
        inputs=[],
        outputs=[pair_plot_output, heatmap_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after the "Visualize Relationships" section

    # Predict Automobile Prices
    gr.Markdown("## Predict Automobile Prices")

    with gr.Row():
        horsepower_input = gr.Slider(label="Horsepower", minimum=50, maximum=300, step=10, value=150)
        curb_weight_input = gr.Slider(label="Curb Weight (lbs)", minimum=1500, maximum=4000, step=100, value=2500)
        engine_size_input = gr.Slider(label="Engine Size (cc)", minimum=50, maximum=300, step=10, value=120)
        highway_mpg_input = gr.Slider(label="Highway MPG", minimum=10, maximum=50, step=5, value=30)

    with gr.Row():
        predicted_price_output = gr.Textbox(label="Predicted Price (in dollars)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)

    with gr.Row():
        actual_vs_predicted_output = gr.Image(label="Actual vs Predicted Plot")
        residual_plot_output = gr.Image(label="Residual Plot")

    predict_button = gr.Button("Predict and Visualize")

    predict_button.click(
        predict_and_visualize,
        inputs=[horsepower_input, curb_weight_input, engine_size_input, highway_mpg_input],
        outputs=[predicted_price_output, r2_output, mse_output, actual_vs_predicted_output, residual_plot_output]
    )

    gr.Markdown("<hr>")  # Add a horizontal line after the "Prediction and Results" section

# Launch the app
demo.launch(share=True)
