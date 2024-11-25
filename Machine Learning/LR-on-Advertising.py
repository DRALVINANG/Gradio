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

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
# Load the Advertising dataset from the provided URL
url = 'https://www.alvinang.sg/s/Advertising.csv'
advert = pd.read_csv(url)

# Define X (features) and y (target)
X = advert[['TV']]
y = advert['Sales']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model parameters (intercept and coefficient)
intercept = model.intercept_
coefficient = model.coef_[0]

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
    sns.pairplot(advert[['TV', 'Sales']])
    pair_plot_path = "pair_plot.png"
    plt.savefig(pair_plot_path)
    plt.close()

    # Generate correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(advert[['TV', 'Sales']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    return pair_plot_path, heatmap_path

# Function to make predictions and visualize results
def predict_and_visualize(tv_budget):
    # Predict sales for the input TV advertising budget
    predicted_sales = model.predict([[tv_budget]])[0]

    # Create a plot showing the regression line and input prediction
    plt.figure(figsize=(12, 6))
    
    # Scatter plot showing actual training data
    plt.scatter(X_train, y_train, label='Training Data', color='blue')
    
    # Regression line for training data
    sales_pred_train = model.predict(X_train)
    plt.plot(X_train, sales_pred_train, 'r', label='Regression Line (Train)', linewidth=2)
    
    # Highlight the input prediction
    plt.scatter([tv_budget], [predicted_sales], color='orange', label=f'Prediction: {predicted_sales:.2f} units', s=100)
    
    # Add labels, title, and legend for clarity
    plt.xlabel('TV Advertising Costs ($)')
    plt.ylabel('Sales (units)')
    plt.title('TV Advertising vs Sales (Train-Test Split)')
    plt.legend()
    
    # Save the plot to display in Gradio
    plot_path = "regression_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Explain model performance
    mse_comment = "Good Fit" if mse <= 10 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"

    return predicted_sales, f"{r2:.2f} ({r2_comment})", f"{mse:.2f} ({mse_comment})", plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# TV Advertising vs Sales - Linear Regression")
    gr.Markdown("""
    ## About the Dataset:
    This dataset consists of sales data for a product across 200 different markets. It includes the **TV advertising budget** and the corresponding **sales** for each market.

    - **TV Advertising Budget (TV)**: Advertising budget spent on TV (in $).
    - **Sales**: The number of units sold in each market (target variable).

    **Dataset Summary**:
    - **200 rows** (markets) and **2 variables**:
        - TV: Advertising budget for TV (numeric, feature).
        - Sales: Product sales (numeric, target variable).
    """)

    gr.Markdown("### Dataset Preview:")
    dataset_preview = gr.Dataframe(label="Dataset Preview", value=advert.head())

    gr.Markdown("""
    ## Visualize Relationships:
    Below are visualizations to better understand the dataset:
    - **Pair Plot**: Shows the relationship between TV advertising budgets and sales.
    - **Correlation Heatmap**: Displays the correlation coefficient between TV and sales.
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

    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
    
    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", interactive=False)
        mse_output = gr.Textbox(label="Mean Squared Error (MSE)", interactive=False)
    
    plot_output = gr.Image(label="Regression Plot")

    # Button to make predictions and visualize results
    predict_button = gr.Button("Predict and Visualize")
    predict_button.click(predict_and_visualize, inputs=tv_budget_input, outputs=[predicted_sales_output, r2_output, mse_output, plot_output])

# Launch the app with share=True for Colab
demo.launch(share=True)
