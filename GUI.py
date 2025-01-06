import mlflow
import pandas as pd
import streamlit as st

# Set the tracking URI to your DagsHub MLflow instance
mlflow.set_tracking_uri("https://dagshub.com/prathamesh.khade20/Watre_test.mlflow")  # URL to track the experiment

# Specify the model name
model_name = "Best Model"  # Registered model name

# Default values for the input fields
default_values = {
    'ph': 3.71608,
    'Hardness': 204.89045,
    'Solids': 20791.318981,
    'Chloramines': 7.300212,
    'Sulfate': 368.516441,
    'Conductivity': 564.308654,
    'Organic_carbon': 10.379783,
    'Trihalomethanes': 86.99097,
    'Turbidity': 2.963135
}

# Load the model
def load_model():
    try:
        # Create an MlflowClient to interact with the MLflow server
        client = mlflow.tracking.MlflowClient()
        # Get the latest version of the model in the Production stage
        versions = client.get_latest_versions(model_name)

        if versions:
            latest_version = versions[0].version
            run_id = versions[0].run_id  # Fetching the run ID from the latest version

            # Construct the logged_model string
            logged_model = f'runs:/{run_id}/{model_name}'
            # Load the model
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            st.write(f"Model loaded from {logged_model}")  # Debug message
            return loaded_model
        else:
            st.error("No model found in the 'Production' stage.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to validate input data
def validate_input(input_value, field_name):
    try:
        return float(input_value)
    except ValueError:
        st.error(f"Invalid input for {field_name}. Please enter a numeric value.")
        return None

# Function to make predictions
def make_prediction(model):
    try:
        # Validate and collect input data from Streamlit form
        input_data = {
            'ph': [validate_input(st.session_state.pH, "pH") or default_values['ph']],
            'Hardness': [validate_input(st.session_state.Hardness, "Hardness") or default_values['Hardness']],
            'Solids': [validate_input(st.session_state.Solids, "Solids") or default_values['Solids']],
            'Chloramines': [validate_input(st.session_state.Chloramines, "Chloramines") or default_values['Chloramines']],
            'Sulfate': [validate_input(st.session_state.Sulfate, "Sulfate") or default_values['Sulfate']],
            'Conductivity': [validate_input(st.session_state.Conductivity, "Conductivity") or default_values['Conductivity']],
            'Organic_carbon': [validate_input(st.session_state.Organic_carbon, "Organic_carbon") or default_values['Organic_carbon']],
            'Trihalomethanes': [validate_input(st.session_state.Trihalomethanes, "Trihalomethanes") or default_values['Trihalomethanes']],
            'Turbidity': [validate_input(st.session_state.Turbidity, "Turbidity") or default_values['Turbidity']]
        }

        # Check if any input is invalid
        if None in input_data.values():
            return  # Do not proceed with prediction if any input is invalid

        # Convert input data to DataFrame
        data = pd.DataFrame(input_data)

        if model is not None:
            # Make prediction
            prediction = model.predict(data)

            # Display prediction result
            if prediction[0] == 1:  # Assuming 1 indicates potable
                st.success("Water is potable.")
            else:  # Assuming 0 indicates not potable
                st.error("Water is not potable.")
            st.write(f"Prediction result: {prediction[0]}")  # Debug message
        else:
            st.error("Model not loaded.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write(f"Error during prediction: {e}")  # Debug message

# Main function to run Streamlit app
def main():
    st.title("Water Quality Prediction")

    # Check if the model is already loaded, if not load it
    if "model" not in st.session_state:
        st.session_state.model = load_model()

    # User inputs for water quality prediction with default values
    st.text_input("pH", key="pH", value=default_values['ph'])
    st.text_input("Hardness", key="Hardness", value=default_values['Hardness'])
    st.text_input("Solids", key="Solids", value=default_values['Solids'])
    st.text_input("Chloramines", key="Chloramines", value=default_values['Chloramines'])
    st.text_input("Sulfate", key="Sulfate", value=default_values['Sulfate'])
    st.text_input("Conductivity", key="Conductivity", value=default_values['Conductivity'])
    st.text_input("Organic_carbon", key="Organic_carbon", value=default_values['Organic_carbon'])
    st.text_input("Trihalomethanes", key="Trihalomethanes", value=default_values['Trihalomethanes'])
    st.text_input("Turbidity", key="Turbidity", value=default_values['Turbidity'])

    # Prediction button
    if st.button("Predict"):
        make_prediction(st.session_state.model)

# Run Streamlit app
if __name__ == "__main__":
    main()
