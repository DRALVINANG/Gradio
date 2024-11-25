# Install necessary libraries
!pip install gradio seaborn matplotlib scikit-learn pandas

# Import necessary libraries
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Advertising dataset from the provided URL
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

# Get the model parameters (intercept and coefficients)
intercept = model.intercept_
coefficients = list(zip(X.columns, model.coef_))

# Predict values for the test set
sales_pred_test = model.predict(X_test)

# Calculate the R-squared value and Mean Squared Error
r2 = r2_score(y_test, sales_pred_test)
mse = mean_squared_error(y_test, sales_pred_test)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
# Function to generate and save pair plot and correlation heatmap
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

# Function to make predictions and visualize results
def predict_and_visualize(tv_budget, radio_budget, newspaper_budget):
    # Predict sales for the input budgets
    predicted_sales = model.predict([[tv_budget, radio_budget, newspaper_budget]])[0]

    # Create a residual plot
    plt.figure(figsize=(12, 6))
    sns.residplot(x=y_test, y=sales_pred_test, lowess=True, color="g")
    plt.xlabel('Actual Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Plot: Actual Sales vs Predicted Sales')
    residual_plot_path = "residual_plot.png"
    plt.savefig(residual_plot_path)
    plt.close()

    # Create a 3D scatter plot for TV, Radio, and Sales
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(advert['TV'], advert['Radio'], advert['Sales'], color='b')
    ax.set_xlabel('TV Advertising Costs ($)')
    ax.set_ylabel('Radio Advertising Costs ($)')
    ax.set_zlabel('Sales (units)')
    plt.title('3D Plot of TV, Radio Advertising vs Sales')
    scatter_plot_path = "scatter_plot_3d.png"
    plt.savefig(scatter_plot_path)
    plt.close()

    # Explain model performance
    mse_comment = "Good Fit" if mse <= 10 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"

    return predicted_sales, f"{r2:.2f} ({r2_comment})", f"{mse:.2f} ({mse_comment})", residual_plot_path, scatter_plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
# Create a Gradio app interface
with gr.Blocks() as demo:
    gr.Markdown("# Advertising Campaigns vs Sales - Multiple Linear Regression")
    gr.Markdown("""
    ## About the Dataset:
    The **Advertising dataset** consists of sales data for a product across 200 different markets. It includes advertising budgets allocated to three types of media:
    - **TV**: Advertising budget for TV (in $).
    - **Radio**: Advertising budget for Radio (in $).
    - **Newspaper**: Advertising budget for Newspapers (in $).

    The target variable, **Sales**, represents the sales of the product in units for each market.

    **Dataset Summary**:
    - **200 rows** (markets) and **4 variables**:
        - TV: Advertising budget for TV (numeric).
        - Radio: Advertising budget for Radio (numeric).
        - Newspaper: Advertising budget for Newspapers (numeric).
        - Sales: Product sales (numeric, target variable).
    """)

    gr.Markdown("""
    ## Visualize Relationships:
    Below are visualizations to better understand the dataset:
    - Pair Plot: Shows relationships between all variables.
    - Correlation Heatmap: Shows correlation coefficients between variables.
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
    1. Adjust the **advertising budgets** (TV, Radio, Newspaper) using the sliders below.
    2. The app will predict the **sales** and display performance metrics:
        - **R² Score (R-squared):** Measures how well the model fits the data. Higher values (closer to 1) indicate a better fit.
        - **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. Lower values (close to 0) indicate a better fit.
    """)

    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
        radio_budget_input = gr.Slider(label="Radio Advertising Budget ($)", minimum=0, maximum=500, step=10, value=50)
        newspaper_budget_input = gr.Slider(label="Newspaper Advertising Budget ($)", minimum=0, maximum=500, step=10, value=20)
    
    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)
    
    with gr.Row():
        residual_plot_output = gr.Image(label="Residual Plot")
        scatter_plot_output = gr.Image(label="3D Scatter Plot")

    # Button to make predictions and visualize results
    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(
        predict_and_visualize, 
        inputs=[tv_budget_input, radio_budget_input, newspaper_budget_input],
        outputs=[predicted_sales_output, r2_output, mse_output, residual_plot_output, scatter_plot_output]
    )

# Launch the app with share=True for Colab
demo.launch(share=True)
