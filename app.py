import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# Streamlit app configuration
st.set_page_config(
    page_title="Creep Rupture Life Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        padding: 1rem;
        background-color: #fff3f3;
        border-radius: 0.5rem;
        border: 2px solid #ff6b6b;
        margin: 1rem 0;
    }
    .model-selector {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the RegressionNN class
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rates, activation='ReLU'):
        super(RegressionNN, self).__init__()
        
        # Choose activation function
        if activation == 'ReLU':
            act_fn = nn.ReLU
        elif activation == 'LeakyReLU':
            act_fn = nn.LeakyReLU
        elif activation == 'ELU':
            act_fn = nn.ELU
        elif activation == 'GELU':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_models_and_data():
    """Load both models and feature information"""
    try:
        # Load the original unscaled data for feature ranges
        original_data_path = "./Data/data2_cleaned_new_.json"
        
        if not os.path.exists(original_data_path):
            st.error(f"Original data file not found at {original_data_path}")
            return None, None, None, None, None
            
        # Load original data for UI ranges
        with open(original_data_path, "r") as f:
            original_data = json.load(f)
        df_original = pd.DataFrame(original_data)
        
        feature_names = [col for col in df_original.columns if col != "Creep rupture life"]
        feature_stats = df_original[feature_names].describe()
        
        # Load PyTorch Neural Network Model
        pytorch_model = None
        scaling_info = None
        
        try:
            # Load pre-trained scalers from the AdvancedModule_Final directory
            
            minmax_scaler_path = "./Data/minmax_scaler.pkl"
            log_params_path = "./Data/log_scaling_params.pkl"
            
            # Load the pre-fitted MinMax scaler
            with open(minmax_scaler_path, 'rb') as f:
                feature_scaler = pickle.load(f)
            
            # Load the log scaling parameters
            with open(log_params_path, 'rb') as f:
                log_params = pickle.load(f)
            
            creep_life_log_min = log_params['min_log_value']
            creep_life_log_max = log_params['max_log_value']
            
            # Load the PyTorch model
            model_path = "./NN/corrected_model_20250928_173833.pth"
            params_path = "./NN/corrected_params_20250928_173833.json"
            
            if os.path.exists(model_path) and os.path.exists(params_path):
                # Load parameters
                with open(params_path, 'r') as f:
                    best_params = json.load(f)
                
                # Extract architecture from parameters
                n_layers = best_params['n_layers']
                hidden_layers = [best_params[f'layer_{i}_size'] for i in range(n_layers)]
                dropout_rates = [best_params[f'dropout_{i}'] for i in range(n_layers)]
                
                # Create model with correct architecture
                pytorch_model = RegressionNN(
                    input_dim=20,
                    hidden_layers=hidden_layers,
                    dropout_rates=dropout_rates,
                    activation=best_params['activation']
                )
                
                # Load trained weights
                pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                pytorch_model.eval()
                
                # Create scaling info dictionary
                scaling_info = {
                    'feature_scaler': feature_scaler,
                    'creep_life_log_min': creep_life_log_min,
                    'creep_life_log_max': creep_life_log_max
                }
                
                st.success("✅ PyTorch Neural Network model loaded successfully")
            else:
                st.warning("⚠️ PyTorch model files not found")
        except Exception as e:
            st.warning(f"⚠️ Error loading PyTorch model: {str(e)}")
        
        # Load AutoGluon Model
        autogluon_model = None
        
        try:
            autogluon_path = "./AutoML/ag-20250930_130252"
            
            if os.path.exists(autogluon_path):
                autogluon_model = TabularPredictor.load(autogluon_path, require_py_version_match=False)
                st.success("✅ AutoGluon model loaded successfully")
            else:
                st.warning("⚠️ AutoGluon model not found")
        except Exception as e:
            st.warning(f"⚠️ Error loading AutoGluon model: {str(e)}")
        
        return pytorch_model, autogluon_model, feature_names, feature_stats, scaling_info
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def create_input_form(feature_names, feature_stats):
    """Create input form for model features using original (unscaled) ranges"""
    st.sidebar.header("🔧 Input Parameters")
    
    inputs = {}
    
    # Chemical composition section with constraint
    st.sidebar.subheader("Chemical Composition (%)")
    st.sidebar.info("⚠️ Chemical elements must sum to 100%")
    
    chemical_features = ['Ni', 'Cr', 'Co', 'Fe', 'Al', 'Ti', 'Nb', 'Mo', 'W', 'C', 'B', 'Zr']
    available_chemical_features = [f for f in chemical_features if f in feature_names]
    
    # Initialize chemical composition with mean values
    if 'chemical_composition' not in st.session_state:
        st.session_state.chemical_composition = {
            feature: float(feature_stats.loc['mean', feature]) 
            for feature in available_chemical_features
        }
        # Normalize to sum to 100%
        total = sum(st.session_state.chemical_composition.values())
        if total > 0:
            for feature in available_chemical_features:
                st.session_state.chemical_composition[feature] = (
                    st.session_state.chemical_composition[feature] / total * 100
                )
    
    # Display current total
    current_total = sum(st.session_state.chemical_composition.values())
    
    # Color code the total based on how close it is to 100%
    if abs(current_total - 100) < 0.1:
        total_color = "green"
        total_icon = "✅"
    elif abs(current_total - 100) < 1:
        total_color = "orange"
        total_icon = "⚠️"
    else:
        total_color = "red"
        total_icon = "❌"
    
    st.sidebar.markdown(f"**Current Total: {total_icon} <span style='color:{total_color}'>{current_total:.2f}%</span>**", unsafe_allow_html=True)
    
    # Create input fields for chemical composition using actual data ranges
    for feature in available_chemical_features:
        min_val = float(feature_stats.loc['min', feature])
        max_val = float(feature_stats.loc['max', feature])
        current_val = st.session_state.chemical_composition[feature]
        
        # Ensure current value is within bounds
        current_val = max(min_val, min(max_val, current_val))
        st.session_state.chemical_composition[feature] = current_val
        
        new_val = st.sidebar.slider(
            f"{feature} (Range: {min_val:.3f} - {max_val:.3f}%)",
            min_value=min_val,
            max_value=max_val,
            value=current_val,
            step=max(0.001, (max_val - min_val) / 1000),
            format="%.3f",
            key=f"slider_{feature}"
        )
        
        st.session_state.chemical_composition[feature] = new_val
        inputs[feature] = new_val
    
    # Normalize button with bounds checking
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Normalize to 100%", key="normalize_btn"):
            current_total = sum(st.session_state.chemical_composition.values())
            if current_total > 0:
                for feature in available_chemical_features:
                    min_val = float(feature_stats.loc['min', feature])
                    max_val = float(feature_stats.loc['max', feature])
                    
                    # Calculate normalized value
                    normalized_val = st.session_state.chemical_composition[feature] / current_total * 100
                    
                    # Ensure normalized value stays within data bounds
                    normalized_val = max(min_val, min(max_val, normalized_val))
                    st.session_state.chemical_composition[feature] = normalized_val
                st.rerun()
    
    with col2:
        if st.button("↩️ Reset to Mean", key="reset_btn"):
            for feature in available_chemical_features:
                st.session_state.chemical_composition[feature] = float(feature_stats.loc['mean', feature])
            # Normalize after reset
            current_total = sum(st.session_state.chemical_composition.values())
            if current_total > 0:
                for feature in available_chemical_features:
                    min_val = float(feature_stats.loc['min', feature])
                    max_val = float(feature_stats.loc['max', feature])
                    
                    normalized_val = st.session_state.chemical_composition[feature] / current_total * 100
                    normalized_val = max(min_val, min(max_val, normalized_val))
                    st.session_state.chemical_composition[feature] = normalized_val
            st.rerun()
    
    # Warning if total is not 100%
    if abs(current_total - 100) > 0.1:
        st.sidebar.warning(f"⚠️ Total composition is {current_total:.2f}%. Click 'Normalize to 100%' to adjust proportionally.")
    
    # Additional warning if values are at bounds after normalization
    at_bounds = []
    for feature in available_chemical_features:
        min_val = float(feature_stats.loc['min', feature])
        max_val = float(feature_stats.loc['max', feature])
        current_val = st.session_state.chemical_composition[feature]
        
        if abs(current_val - min_val) < 0.001 or abs(current_val - max_val) < 0.001:
            at_bounds.append(feature)
    
    if at_bounds:
        st.sidebar.info(f"ℹ️ Elements at data bounds: {', '.join(at_bounds)}")
    
    # Test conditions section
    st.sidebar.subheader("Test Conditions")
    test_features = ['Test temperature (℃)', 'Test stress (Mpa)']
    
    for feature in test_features:
        if feature in feature_names:
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            mean_val = float(feature_stats.loc['mean', feature])
            
            inputs[feature] = st.sidebar.slider(
                f"{feature} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=max(1.0, (max_val - min_val) / 100),
                format="%.1f"
            )
    
    # Heat treatment section
    st.sidebar.subheader("Heat Treatment Parameters")
    heat_treatment_features = [
        'solution treatment temperature', 'solution treatment time',
        'Stable aging temperature (℃)', 'Stable aging time (h)',
        'Aging temperature (℃)', 'Aging time (h)'
    ]
    
    for feature in heat_treatment_features:
        if feature in feature_names:
            min_val = float(feature_stats.loc['min', feature])
            max_val = float(feature_stats.loc['max', feature])
            mean_val = float(feature_stats.loc['mean', feature])
            
            inputs[feature] = st.sidebar.slider(
                f"{feature} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=max(0.1, (max_val - min_val) / 100),
                format="%.2f"
            )
    
    return inputs
def scale_inputs(inputs, feature_names, scaling_info):
    """Scale the user inputs to match PyTorch model training data"""
    try:
        # Convert inputs to array in correct order
        input_array = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)
        
        # Scale using the fitted scaler
        scaled_inputs = scaling_info['feature_scaler'].transform(input_array)
        
        return scaled_inputs
    except Exception as e:
        st.error(f"Error scaling inputs: {str(e)}")
        return None

def unscale_prediction(scaled_prediction, scaling_info):
    """Convert scaled prediction back to original units"""
    try:
        # Unscale from [0,1] back to log space
        log_prediction = (scaled_prediction * (scaling_info['creep_life_log_max'] - scaling_info['creep_life_log_min'])) + scaling_info['creep_life_log_min']
        
        # Convert from log space back to original scale
        original_prediction = np.expm1(log_prediction)  # inverse of log1p
        
        return original_prediction
    except Exception as e:
        st.error(f"Error unscaling prediction: {str(e)}")
        return None

def make_pytorch_prediction(model, inputs, feature_names, scaling_info):
    """Make prediction using the PyTorch model"""
    try:
        # Scale inputs
        scaled_inputs = scale_inputs(inputs, feature_names, scaling_info)
        if scaled_inputs is None:
            return None
        
        # Convert to tensor
        input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
        
        # Make prediction (this will be in scaled space)
        with torch.no_grad():
            scaled_prediction = model(input_tensor).item()
        
        # Unscale prediction to original units
        original_prediction = unscale_prediction(scaled_prediction, scaling_info)
        
        return original_prediction
    except Exception as e:
        st.error(f"Error making PyTorch prediction: {str(e)}")
        return None

def make_autogluon_prediction(model, inputs, feature_names, scaling_info):
    """Make prediction using the AutoGluon model"""
    try:
        # Scale inputs first (AutoGluon was trained on scaled data)
        scaled_inputs = scale_inputs(inputs, feature_names, scaling_info)
        if scaled_inputs is None:
            return None
        
        # Create input dataframe with scaled values
        scaled_input_dict = {feature: scaled_inputs[0][i] for i, feature in enumerate(feature_names)}
        input_data = pd.DataFrame([scaled_input_dict])

        best_model_name = "RandomForest_r16_BAG_L2"
        
        # Make prediction (this will be in scaled space)
        scaled_prediction = model.predict(input_data, model = best_model_name)
        prediction_value = scaled_prediction.iloc[0] if hasattr(scaled_prediction, 'iloc') else scaled_prediction[0]
        
        # Unscale prediction to original units
        original_prediction = unscale_prediction(prediction_value, scaling_info)
        
        return original_prediction
    except Exception as e:
        st.error(f"Error making AutoGluon prediction: {str(e)}")
        return None

def main():
    # App header
    st.markdown('<div class="main-header">🔬 Creep Rupture Life Prediction</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models and data
    result = load_models_and_data()
    if result[2] is None:  # Check if feature_names is None
        st.error("Failed to load data. Please check the file paths and try again.")
        return
    
    pytorch_model, autogluon_model, feature_names, feature_stats, scaling_info = result
    
    # Model selection
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_pytorch = st.checkbox("🧠 PyTorch Neural Network", value=pytorch_model is not None, disabled=pytorch_model is None)
    with col2:
        use_autogluon = st.checkbox("🤖 AutoGluon ML", value=autogluon_model is not None, disabled=autogluon_model is None)
    with col3:
        compare_models = st.checkbox("⚖️ Compare Models", value=False, disabled=pytorch_model is None or autogluon_model is None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create sidebar inputs
    inputs = create_input_form(feature_names, feature_stats)
    
    # Main prediction section
    st.header("📊 Model Predictions")
    
    # Make prediction button
    if st.button("🚀 Make Prediction", type="primary"):
        pytorch_pred = None
        autogluon_pred = None
        
        if use_pytorch and pytorch_model is not None:
            pytorch_pred = make_pytorch_prediction(pytorch_model, inputs, feature_names, scaling_info)
            if pytorch_pred is not None:
                st.markdown(
                    f'<div class="prediction-result">🧠 PyTorch NN: {pytorch_pred:.2f} hours</div>',
                    unsafe_allow_html=True
                )
        
        if use_autogluon and autogluon_model is not None:
            autogluon_pred = make_autogluon_prediction(autogluon_model, inputs, feature_names, scaling_info)
            if autogluon_pred is not None:
                st.markdown(
                    f'<div class="prediction-result">🤖 AutoGluon ML: {autogluon_pred:.2f} hours</div>',
                    unsafe_allow_html=True
                )
        
        if compare_models and pytorch_pred is not None and autogluon_pred is not None:
            # Show difference
            diff = abs(pytorch_pred - autogluon_pred)
            st.info(f"📈 Prediction difference: {diff:.2f} hours ({(diff/max(pytorch_pred, autogluon_pred)*100):.1f}%)")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔬 Creep Rupture Life Prediction Platform</p>
            <p>Built with Streamlit, PyTorch, and AutoGluon</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()