# Install necessary libraries
!pip install gradio seaborn matplotlib scikit-learn pandas

# Import necessary libraries
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

# Calculate the R-squared value
r2 = r2_score(y_test, sales_pred_test)

#--------------------------------------------------------------------
# Gradio App Functions
#--------------------------------------------------------------------
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

    return predicted_sales, r2, plot_path

#--------------------------------------------------------------------
# Gradio Interface
#--------------------------------------------------------------------
# Create a Gradio app interface
with gr.Blocks() as demo:
    gr.Markdown("# TV Advertising vs Sales - Linear Regression")
    gr.Markdown("""
    This app allows you to explore the relationship between TV advertising costs and sales.
    Adjust the TV advertising budget below to see the predicted sales and visualize the results.
    """)
    
    with gr.Row():
        tv_budget_input = gr.Slider(label="TV Advertising Budget ($)", minimum=0, maximum=500, step=10, value=100)
    
    with gr.Row():
        predicted_sales_output = gr.Textbox(label="Predicted Sales (units)", interactive=False)
        r2_output = gr.Textbox(label="R-squared Value", value=f"{r2:.2f}", interactive=False)
    
    plot_output = gr.Image(label="Regression Plot")

    # Button to make predictions and visualize results
    predict_button = gr.Button("Predict and Visualize")

    # Connect the function to Gradio inputs and outputs
    predict_button.click(predict_and_visualize, inputs=tv_budget_input, outputs=[predicted_sales_output, r2_output, plot_output])

# Launch the app with share=True for Colab
demo.launch(share=True)
