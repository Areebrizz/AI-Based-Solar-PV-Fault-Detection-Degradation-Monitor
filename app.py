import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Solar PV Fault Detection System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B00;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-radius: 5px;
        color: #cc0000;
    }
    .alert-medium {
        background-color: #ffe6cc;
        padding: 0.5rem;
        border-radius: 5px;
        color: #e68a00;
    }
    .alert-low {
        background-color: #ccffcc;
        padding: 0.5rem;
        border-radius: 5px;
        color: #006600;
    }
</style>
""", unsafe_allow_html=True)

class SolarPVAnalyzer:
    """Main class for Solar PV analysis and fault detection"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.pca = PCA(n_components=2)
        
    def clean_data(self, df):
        """Clean and validate input data"""
        
        # Make a copy
        df_clean = df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['voltage', 'current', 'power', 'temperature', 'irradiance', 'system_capacity']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with reasonable defaults
        for col in numeric_cols:
            if col in df_clean.columns:
                if df_clean[col].isna().any():
                    if col == 'temperature':
                        df_clean[col] = df_clean[col].fillna(25)  # Room temperature
                    elif col == 'irradiance':
                        df_clean[col] = df_clean[col].fillna(500)  # Average irradiance
                    elif col == 'voltage':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif col == 'current':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif col == 'power':
                        # Calculate power from voltage and current if available
                        if all(c in df_clean.columns for c in ['voltage', 'current']):
                            mask = df_clean['power'].isna()
                            df_clean.loc[mask, 'power'] = df_clean.loc[mask, 'voltage'] * df_clean.loc[mask, 'current'] / 1000
                        else:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Ensure positive values for physical quantities
        positive_cols = ['voltage', 'current', 'power', 'irradiance', 'system_capacity']
        for col in positive_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=0.001)
        
        # Ensure temperature is within reasonable bounds
        if 'temperature' in df_clean.columns:
            df_clean['temperature'] = df_clean['temperature'].clip(lower=-20, upper=100)
        
        return df_clean
    
    def compute_performance_metrics(self, df):
        """Calculate IEC 61724 performance metrics"""
        
        # Make a copy
        df_metrics = df.copy()
        
        # Ensure required columns exist
        required_cols = ['irradiance', 'system_capacity', 'power']
        if not all(col in df_metrics.columns for col in required_cols):
            st.error(f"Missing required columns for metrics calculation. Need: {required_cols}")
            return df_metrics
        
        # Performance Ratio (PR)
        # Avoid division by zero
        irradiance_safe = df_metrics['irradiance'].replace(0, 1)
        system_capacity_safe = df_metrics['system_capacity'].replace(0, 1)
        
        expected_power = irradiance_safe * system_capacity_safe * 0.2  # 20% efficiency assumption
        expected_power = expected_power.replace(0, 1)  # Avoid division by zero
        
        df_metrics['performance_ratio'] = (df_metrics['power'] / expected_power).clip(upper=1, lower=0)
        
        # Capacity Factor (CF)
        df_metrics['capacity_factor'] = (df_metrics['power'] / system_capacity_safe).clip(upper=1, lower=0)
        
        # Normalized power
        df_metrics['normalized_power'] = df_metrics['power'] / irradiance_safe
        
        # Temperature corrected output
        temp_coeff = -0.004  # Typical temperature coefficient for silicon PV
        ref_temp = 25
        if 'temperature' in df_metrics.columns:
            df_metrics['temp_corrected_power'] = df_metrics['power'] * (1 + temp_coeff * (df_metrics['temperature'] - ref_temp))
            # Clip to reasonable values
            df_metrics['temp_corrected_power'] = df_metrics['temp_corrected_power'].clip(lower=0)
        else:
            df_metrics['temp_corrected_power'] = df_metrics['power']
        
        return df_metrics
    
    def engineer_features(self, df):
        """Create engineered features for anomaly detection"""
        
        # Make a copy
        df_features = df.copy()
        
        # Time-based features
        if 'timestamp' in df_features.columns:
            try:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
                df_features['hour'] = df_features['timestamp'].dt.hour
                df_features['day_of_year'] = df_features['timestamp'].dt.dayofyear
            except:
                # If timestamp parsing fails, create dummy time features
                df_features['hour'] = np.arange(len(df_features)) % 24
                df_features['day_of_year'] = np.arange(len(df_features)) % 365
        
        # Operational features
        if all(col in df_features.columns for col in ['current', 'voltage']):
            voltage_safe = df_features['voltage'].replace(0, 0.001)
            df_features['current_voltage_ratio'] = df_features['current'] / voltage_safe
            df_features['power_voltage_ratio'] = df_features['power'] / voltage_safe
        else:
            df_features['current_voltage_ratio'] = 1.0
            df_features['power_voltage_ratio'] = 1.0
        
        # Deviation from expected
        if all(col in df_features.columns for col in ['irradiance', 'system_capacity', 'power']):
            irradiance_safe = df_features['irradiance'].replace(0, 1)
            system_capacity_safe = df_features['system_capacity'].replace(0, 1)
            expected_power = irradiance_safe * system_capacity_safe * 0.2
            expected_power = expected_power.replace(0, 0.001)
            df_features['power_deviation'] = (df_features['power'] - expected_power) / expected_power
        else:
            df_features['power_deviation'] = 0
        
        # Rolling statistics (handle NaN values)
        if 'power' in df_features.columns:
            df_features['power_rolling_mean'] = df_features['power'].rolling(window=min(24, len(df_features)), min_periods=1).mean()
            df_features['power_rolling_std'] = df_features['power'].rolling(window=min(24, len(df_features)), min_periods=1).std()
            
            # Avoid division by zero
            std_safe = df_features['power_rolling_std'].replace(0, 0.001)
            df_features['power_zscore'] = (df_features['power'] - df_features['power_rolling_mean']) / std_safe
        else:
            df_features['power_rolling_mean'] = 0
            df_features['power_rolling_std'] = 0
            df_features['power_zscore'] = 0
        
        # Fill any remaining NaN values
        df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_features
    
    def detect_anomalies(self, df):
        """Train Isolation Forest and detect anomalies"""
        
        # Make a copy
        df_anomalies = df.copy()
        
        # Features for anomaly detection
        feature_columns = [
            'normalized_power', 'temp_corrected_power', 
            'performance_ratio', 'capacity_factor',
            'current_voltage_ratio', 'power_deviation',
            'power_zscore'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in df_anomalies.columns]
        
        if len(available_features) < 3:
            st.warning(f"Insufficient features for anomaly detection. Need at least 3, found {len(available_features)}")
            df_anomalies['anomaly_score'] = 0
            df_anomalies['is_anomaly'] = 0
            df_anomalies['fault_type'] = 'Normal'
            return df_anomalies
        
        # Prepare data
        X = df_anomalies[available_features].copy()
        
        # Check for NaN or infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            st.warning("NaN or infinite values found in features. Cleaning data...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Ensure all values are finite
        if not np.isfinite(X.values).all():
            st.error("Data contains non-finite values that cannot be processed.")
            df_anomalies['anomaly_score'] = 0
            df_anomalies['is_anomaly'] = 0
            df_anomalies['fault_type'] = 'Normal'
            return df_anomalies
        
        try:
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            
            # Predict anomalies
            df_anomalies['anomaly_score'] = self.model.decision_function(X_scaled)
            predictions = self.model.predict(X_scaled)
            df_anomalies['is_anomaly'] = np.where(predictions == 1, 0, 1)  # Convert to 0=normal, 1=anomaly
            
            # Fault classification
            df_anomalies['fault_type'] = self.classify_faults(df_anomalies)
            
        except Exception as e:
            st.error(f"Error in anomaly detection: {str(e)}")
            df_anomalies['anomaly_score'] = 0
            df_anomalies['is_anomaly'] = 0
            df_anomalies['fault_type'] = 'Normal'
        
        return df_anomalies
    
    def classify_faults(self, df):
        """Classify detected anomalies into fault types"""
        
        fault_types = []
        
        for idx, row in df.iterrows():
            if row['is_anomaly'] == 0:
                fault_types.append('Normal')
                continue
            
            # Default to Unknown Anomaly
            fault_type = 'Unknown Anomaly'
            
            # Check for required columns
            has_irradiance = 'irradiance' in row
            has_temperature = 'temperature' in row
            has_normalized_power = 'normalized_power' in row
            has_power_deviation = 'power_deviation' in row
            has_current_voltage_ratio = 'current_voltage_ratio' in row
            has_performance_ratio = 'performance_ratio' in row
            has_capacity_factor = 'capacity_factor' in row
            
            # Soiling detection: Low normalized power despite good irradiance
            if (has_normalized_power and has_irradiance and has_temperature and
                row['normalized_power'] < 0.6 and 
                row['irradiance'] > 500 and
                row['temperature'] < 40):
                fault_type = 'Soiling'
                
            # Shading detection: Sudden drops in current with stable voltage
            elif (has_power_deviation and has_current_voltage_ratio and
                  row['power_deviation'] < -0.3 and 
                  abs(row['current_voltage_ratio']) > 2):
                fault_type = 'Shading'
                
            # Inverter degradation: Gradual power loss with normal I-V characteristics
            elif (has_performance_ratio and has_capacity_factor and has_current_voltage_ratio and
                  row['performance_ratio'] < 0.7 and 
                  row['capacity_factor'] < 0.5 and
                  0.9 < row['current_voltage_ratio'] < 1.1):
                fault_type = 'Inverter Degradation'
                
            # Temperature issues
            elif has_temperature and row['temperature'] > 60:
                fault_type = 'Overheating'
                
            # Electrical issues
            elif 'voltage' in row and 'current' in row and (row['voltage'] < 200 or row['current'] < 2):
                fault_type = 'Electrical Fault'
            
            fault_types.append(fault_type)
                
        return fault_types
    
    def calculate_degradation(self, df):
        """Calculate daily and monthly degradation trends"""
        
        # Make copies
        df_degradation = df.copy()
        
        # Initialize with empty dataframes
        daily_perf = pd.DataFrame()
        monthly_perf = pd.DataFrame()
        
        if 'timestamp' in df_degradation.columns and 'performance_ratio' in df_degradation.columns:
            try:
                df_degradation['timestamp'] = pd.to_datetime(df_degradation['timestamp'])
                df_degradation['date'] = df_degradation['timestamp'].dt.date
                df_degradation['month'] = df_degradation['timestamp'].dt.to_period('M')
                
                # Daily performance
                daily_perf = df_degradation.groupby('date').agg({
                    'performance_ratio': 'mean',
                    'power': 'sum',
                    'irradiance': 'mean'
                }).reset_index()
                
                # Monthly degradation
                monthly_perf = df_degradation.groupby('month').agg({
                    'performance_ratio': 'mean',
                    'capacity_factor': 'mean'
                }).reset_index()
                
                # Calculate degradation rate
                if len(monthly_perf) > 1:
                    monthly_perf['degradation_rate'] = monthly_perf['performance_ratio'].pct_change() * 100
                    monthly_perf['degradation_rate'] = monthly_perf['degradation_rate'].fillna(0)
            
            except Exception as e:
                st.warning(f"Could not calculate degradation trends: {str(e)}")
        
        return daily_perf, monthly_perf
    
    def calculate_fault_severity(self, df):
        """Calculate fault severity index"""
        
        if 'is_anomaly' not in df.columns or df['is_anomaly'].sum() == 0:
            return 0
        
        severity_scores = []
        
        anomalies_df = df[df['is_anomaly'] == 1]
        
        for _, row in anomalies_df.iterrows():
            base_score = abs(row.get('anomaly_score', 0))
            
            # Adjust based on fault type
            fault_type = row.get('fault_type', 'Unknown Anomaly')
            if fault_type == 'Inverter Degradation':
                severity = base_score * 1.5
            elif fault_type == 'Electrical Fault':
                severity = base_score * 1.3
            elif fault_type == 'Soiling':
                severity = base_score * 1.1
            elif fault_type == 'Shading':
                severity = base_score * 1.0
            else:
                severity = base_score
            
            # Adjust based on power loss if available
            if 'power_deviation' in row:
                power_loss = row['power_deviation'] * 100
                severity *= (1 + abs(power_loss) / 100)
            
            severity_scores.append(min(severity, 10))  # Cap at 10
        
        return np.mean(severity_scores) if severity_scores else 0

def generate_synthetic_data(days=30, samples_per_day=24):
    """Generate synthetic PV system data"""
    
    np.random.seed(42)
    timestamps = []
    data = []
    
    start_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        for hour in range(samples_per_day):
            timestamp = start_date + timedelta(days=day, hours=hour)
            
            # Base patterns
            hour_of_day = hour
            day_of_year = day
            
            # Irradiance pattern (sinusoidal with noise)
            irradiance = max(0, 1000 * np.sin(np.pi * hour_of_day / 24) + 
                            np.random.normal(0, 50))
            
            # Temperature pattern
            temperature = 25 + 15 * np.sin(np.pi * hour_of_day / 24) + \
                         np.random.normal(0, 3)
            
            # Voltage (slightly decreasing with temperature)
            voltage = 500 - 0.5 * temperature + np.random.normal(0, 10)
            
            # Current (proportional to irradiance with degradation over time)
            current_base = irradiance / 1000 * 10
            degradation = 1 - (day_of_year * 0.0001)  # 0.01% daily degradation
            current = current_base * degradation + np.random.normal(0, 0.2)
            
            # Power
            power = voltage * current / 1000  # Convert to kW
            
            # Introduce some anomalies
            if day == 15 and 10 <= hour_of_day <= 14:
                # Soiling event
                power *= 0.6
                current *= 0.6
                
            if day == 20 and 13 <= hour_of_day <= 15:
                # Shading event
                power *= 0.4
                voltage *= 0.9
                
            if day >= 25:
                # Gradual inverter degradation
                power *= 0.85
                voltage *= 0.95
            
            data.append({
                'timestamp': timestamp,
                'voltage': max(0, voltage),
                'current': max(0, current),
                'power': max(0, power),
                'temperature': temperature,
                'irradiance': max(0, irradiance),
                'system_capacity': 100  # 100 kW system
            })
    
    df = pd.DataFrame(data)
    
    # Ensure no NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar PV Fault Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/solar-panel.png", 
                width=100)
        st.markdown("### Configuration")
        
        analysis_days = st.slider("Analysis Period (days)", 7, 90, 30)
        contamination = st.slider("Anomaly Detection Sensitivity", 0.01, 0.3, 0.1, 0.01)
        
        st.markdown("---")
        st.markdown("### System Parameters")
        system_capacity = st.number_input("System Capacity (kW)", 10, 1000, 100, step=10)
        location = st.selectbox("Location", ["Desert", "Temperate", "Coastal", "Urban"])
        
        st.markdown("---")
        st.markdown("#### Data Source")
        data_source = st.radio("Choose data source:", 
                              ["Use Synthetic Data", "Upload CSV"])
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload PV data CSV", 
                                            type=['csv'])
        
# Replace the current credits section (lines 428-432) with this:

st.markdown("---")
st.markdown("### 🏆 Developed By")

# Create a professional profile card
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://img.icons8.com/color/96/000000/user-male-circle--v1.png", 
             width=80)
with col2:
    st.markdown("**Areeb Rizwan**")
    st.markdown("⚡ Renewable Energy Data Scientist")
    st.markdown("📍 AI & ML Solutions for Clean Energy")

st.markdown("---")

# Links with icons
st.markdown("### 🌐 Connect With Me")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 18px; font-weight: bold;">📍 Portfolio</div>
            <div style="font-size: 14px;">www.areebrizwan.com</div>
        </div>
        <a href="http://www.areebrizwan.com" target="_blank" style="color: white; text-decoration: none;">
            <div style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 5px; font-size: 14px;">
                Visit →
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")  # Spacer

st.markdown("""
<div style="background: linear-gradient(135deg, #0077b5 0%, #00a0dc 100%); padding: 20px; border-radius: 10px; color: white;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 18px; font-weight: bold;">💼 LinkedIn</div>
            <div style="font-size: 14px;">Professional Profile</div>
        </div>
        <a href="https://www.linkedin.com/in/areebrizwan" target="_blank" style="color: white; text-decoration: none;">
            <div style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 5px; font-size: 14px;">
                Connect →
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Optional: Add GitHub if you have
st.markdown("")  # Spacer
st.markdown("""
<div style="background: linear-gradient(135deg, #333 0%, #666 100%); padding: 20px; border-radius: 10px; color: white;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 18px; font-weight: bold;">💻 GitHub</div>
            <div style="font-size: 14px;">Open Source Projects</div>
        </div>
        <a href="https://github.com/areebrizwan" target="_blank" style="color: white; text-decoration: none;">
            <div style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 5px; font-size: 14px;">
                Follow →
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = SolarPVAnalyzer()
    analyzer.model = IsolationForest(contamination=contamination, random_state=42)
    
    # Load data
    try:
        if data_source == "Use Synthetic Data":
            with st.spinner("Generating synthetic data..."):
                df = generate_synthetic_data(days=analysis_days)
            st.success(f"Synthetic data generated for {analysis_days} days")
            
        elif uploaded_file is not None:
            with st.spinner("Loading uploaded data..."):
                df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            
            # Check for required columns
            required_cols = ['timestamp', 'voltage', 'current', 'power', 
                            'temperature', 'irradiance']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns in uploaded data: {missing_cols}")
                st.info("The system will try to work with available data. Synthetic data may give better results.")
        else:
            st.info("Please upload a CSV file or use synthetic data to begin analysis.")
            
            # Show sample data structure
            st.markdown("### Expected Data Format")
            st.markdown("""
            Your CSV should contain these columns:
            - `timestamp`: Date/time of measurement
            - `voltage`: DC voltage in volts
            - `current`: DC current in amps
            - `power`: Power output in kW
            - `temperature`: Module temperature in °C
            - `irradiance`: Solar irradiance in W/m²
            """)
            
            # Generate and show sample data
            sample_df = generate_synthetic_data(days=2)
            st.dataframe(sample_df.head(10), use_container_width=True)
            
            return
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using synthetic data instead.")
        df = generate_synthetic_data(days=analysis_days)
    
    # Add system capacity to data
    df['system_capacity'] = system_capacity
    
    # Process data with progress indicators
    progress_bar = st.progress(0)
    
    with st.spinner("Cleaning and validating data..."):
        df = analyzer.clean_data(df)
        progress_bar.progress(25)
    
    with st.spinner("Calculating performance metrics..."):
        df = analyzer.compute_performance_metrics(df)
        progress_bar.progress(50)
    
    with st.spinner("Engineering features for analysis..."):
        df = analyzer.engineer_features(df)
        progress_bar.progress(75)
    
    with st.spinner("Running anomaly detection..."):
        df = analyzer.detect_anomalies(df)
        progress_bar.progress(90)
    
    with st.spinner("Calculating trends and severity..."):
        daily_perf, monthly_perf = analyzer.calculate_degradation(df)
        fault_severity = analyzer.calculate_fault_severity(df)
        progress_bar.progress(100)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pr_mean = df['performance_ratio'].mean() if 'performance_ratio' in df.columns else 0
        st.metric("Performance Ratio", 
                 f"{pr_mean:.2%}",
                 f"{((pr_mean - 0.8)/0.8*100):+.1f}%" if pr_mean > 0 else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        cf_mean = df['capacity_factor'].mean() if 'capacity_factor' in df.columns else 0
        st.metric("Capacity Factor", 
                 f"{cf_mean:.2%}",
                 f"{((cf_mean - 0.2)/0.2*100):+.1f}%" if cf_mean > 0 else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
        st.metric("Anomalies Detected", 
                 f"{int(anomalies)}",
                 f"{anomalies/len(df)*100:.1f}%" if len(df) > 0 else "0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        severity_class = "High" if fault_severity > 7 else "Medium" if fault_severity > 4 else "Low"
        severity_color = "#ffcccc" if fault_severity > 7 else "#ffe6cc" if fault_severity > 4 else "#ccffcc"
        st.metric("Fault Severity Index", 
                 f"{fault_severity:.2f}/10",
                 severity_class)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Overview", 
        "🚨 Fault Analysis", 
        "📈 Degradation Trends",
        "🔧 Maintenance Dashboard"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Power vs Irradiance
            if all(col in df.columns for col in ['irradiance', 'power', 'is_anomaly']):
                fig = px.scatter(df, x='irradiance', y='power', 
                                color=df['is_anomaly'].astype(str),
                                title='Power vs Irradiance',
                                labels={'color': 'Anomaly'},
                                color_discrete_map={'0': 'blue', '1': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for Power vs Irradiance plot")
            
            # Performance Ratio Distribution
            if 'performance_ratio' in df.columns:
                fig = px.histogram(df, x='performance_ratio', 
                                  title='Performance Ratio Distribution',
                                  nbins=50)
                fig.add_vline(x=df['performance_ratio'].mean(), 
                             line_dash="dash", 
                             line_color="red",
                             annotation_text=f"Mean: {df['performance_ratio'].mean():.2%}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time series of key metrics
            if 'timestamp' in df.columns and 'power' in df.columns:
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=('Power Generation', 'Performance Ratio'))
                
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['power'], 
                              mode='lines', name='Power'),
                    row=1, col=1
                )
                
                if 'performance_ratio' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['performance_ratio'], 
                                  mode='lines', name='PR'),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fault type distribution
            if 'fault_type' in df.columns and 'is_anomaly' in df.columns:
                fault_counts = df[df['is_anomaly'] == 1]['fault_type'].value_counts()
                if len(fault_counts) > 0:
                    fig = px.pie(values=fault_counts.values, 
                                names=fault_counts.index,
                                title='Fault Type Distribution',
                                hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies detected in the data")
            else:
                st.warning("Fault analysis data not available")
            
            # Anomaly scores distribution
            if 'anomaly_score' in df.columns and 'is_anomaly' in df.columns:
                anomaly_data = df[df['is_anomaly'] == 1]
                if len(anomaly_data) > 0:
                    fig = px.histogram(anomaly_data, 
                                      x='anomaly_score',
                                      title='Anomaly Scores Distribution',
                                      nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fault details table
            if all(col in df.columns for col in ['is_anomaly', 'fault_type', 'anomaly_score']):
                anomalies_df = df[df['is_anomaly'] == 1].copy()
                if not anomalies_df.empty:
                    # Fault timeline
                    if 'timestamp' in anomalies_df.columns:
                        anomalies_df['hour'] = pd.to_datetime(anomalies_df['timestamp']).dt.hour
                        
                        fig = px.density_heatmap(anomalies_df, 
                                                x='timestamp', 
                                                y='hour',
                                                z='anomaly_score',
                                                title='Fault Occurrence Timeline',
                                                color_continuous_scale='Reds')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Fault details table
                    st.subheader("Detected Faults Details")
                    display_cols = ['timestamp', 'fault_type', 'anomaly_score']
                    if 'power' in anomalies_df.columns:
                        display_cols.append('power')
                    if 'performance_ratio' in anomalies_df.columns:
                        display_cols.append('performance_ratio')
                    
                    available_cols = [col for col in display_cols if col in anomalies_df.columns]
                    fault_details = anomalies_df[available_cols].copy()
                    
                    if 'timestamp' in fault_details.columns:
                        fault_details['timestamp'] = pd.to_datetime(fault_details['timestamp'])
                    
                    if 'anomaly_score' in fault_details.columns:
                        fault_details = fault_details.sort_values('anomaly_score', ascending=False)
                    
                    st.dataframe(fault_details.head(20), use_container_width=True)
                else:
                    st.success("✅ No anomalies detected in the dataset!")
            else:
                st.warning("Anomaly detection results not available")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily performance
            if not daily_perf.empty and 'performance_ratio' in daily_perf.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                        y=daily_perf['performance_ratio'],
                                        mode='lines+markers',
                                        name='Performance Ratio',
                                        line=dict(color='blue')))
                
                if 'irradiance' in daily_perf.columns:
                    fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                            y=daily_perf['irradiance']/1000,
                                            mode='lines',
                                            name='Irradiance (kW/m²)',
                                            yaxis='y2',
                                            line=dict(color='orange', dash='dash')))
                
                fig.update_layout(
                    title='Daily Performance Trend',
                    yaxis=dict(title='Performance Ratio'),
                    hovermode='x unified'
                )
                
                if 'irradiance' in daily_perf.columns:
                    fig.update_layout(
                        yaxis2=dict(title='Irradiance (kW/m²)', 
                                   overlaying='y', 
                                   side='right')
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Daily performance data not available")
        
        with col2:
            # Monthly degradation
            if not monthly_perf.empty and 'performance_ratio' in monthly_perf.columns:
                fig = make_subplots(rows=2, cols=1,
                                   subplot_titles=('Monthly Performance', 
                                                  'Degradation Rate'))
                
                fig.add_trace(
                    go.Bar(x=monthly_perf['month'].astype(str), 
                          y=monthly_perf['performance_ratio'],
                          name='Performance Ratio'),
                    row=1, col=1
                )
                
                if 'degradation_rate' in monthly_perf.columns:
                    fig.add_trace(
                        go.Scatter(x=monthly_perf['month'].astype(str), 
                                  y=monthly_perf['degradation_rate'],
                                  mode='lines+markers',
                                  name='Degradation Rate',
                                  line=dict(color='red')),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Maintenance Priority Dashboard")
        
        # Calculate maintenance priorities
        if all(col in df.columns for col in ['is_anomaly', 'fault_type', 'anomaly_score']):
            anomalies_df = df[df['is_anomaly'] == 1]
            
            if not anomalies_df.empty:
                fault_summary = anomalies_df.groupby('fault_type').agg({
                    'anomaly_score': 'mean',
                    'is_anomaly': 'count'
                }).reset_index()
                
                fault_summary.columns = ['Fault Type', 'Avg Severity', 'Occurrences']
                
                if 'power' in anomalies_df.columns and 'power' in df.columns:
                    fault_summary['Avg Power Loss'] = anomalies_df.groupby('fault_type')['power'].mean()
                    fault_summary['Power Loss %'] = (fault_summary['Avg Power Loss'] / df['power'].mean()) * 100
                
                # Assign priority scores
                def assign_priority(row):
                    score = (abs(row['Avg Severity']) * 0.4 + 
                            (row['Occurrences'] / len(df)) * 100 * 0.3)
                    
                    if 'Power Loss %' in row:
                        score += row['Power Loss %'] * 0.3
                    
                    if score > 7:
                        return ('High', '🔴')
                    elif score > 4:
                        return ('Medium', '🟡')
                    else:
                        return ('Low', '🟢')
                
                fault_summary[['Priority', 'Priority Icon']] = fault_summary.apply(
                    assign_priority, axis=1, result_type='expand'
                )
                
                fault_summary = fault_summary.sort_values('Avg Severity', ascending=False)
                
                # Display maintenance table
                st.dataframe(fault_summary, use_container_width=True)
                
                # Maintenance recommendations
                st.subheader("Recommended Actions")
                
                recommendations = {
                    'Soiling': "• Schedule panel cleaning\n• Inspect cleaning system\n• Check weather patterns",
                    'Shading': "• Trim surrounding vegetation\n• Review panel positioning\n• Consider micro-inverters",
                    'Inverter Degradation': "• Schedule inverter inspection\n• Check warranty status\n• Plan for replacement",
                    'Electrical Fault': "• Immediate electrical inspection\n• Check connections and wiring\n• Review safety systems",
                    'Overheating': "• Improve ventilation\n• Check cooling systems\n• Review installation location",
                    'Unknown Anomaly': "• Investigate further\n• Check all system components\n• Review historical data"
                }
                
                for fault_type in fault_summary['Fault Type'].unique():
                    if fault_type in recommendations:
                        priority = fault_summary[fault_summary['Fault Type'] == fault_type]['Priority'].iloc[0]
                        
                        if priority == 'High':
                            st.markdown(f'<div class="alert-high">', unsafe_allow_html=True)
                        elif priority == 'Medium':
                            st.markdown(f'<div class="alert-medium">', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert-low">', unsafe_allow_html=True)
                        
                        st.markdown(f"**{fault_type}** - {priority} Priority")
                        st.markdown(recommendations[fault_type])
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")
            else:
                st.success("✅ No maintenance required - no anomalies detected!")
        
        # Performance loss estimation
        if 'power_deviation' in df.columns and 'is_anomaly' in df.columns:
            total_power_loss = df[df['is_anomaly'] == 1]['power_deviation'].sum() * 100
            if total_power_loss != 0:
                st.metric("Estimated Total Performance Loss", 
                         f"{abs(total_power_loss):.1f}%",
                         delta_color="inverse")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check your data and try again. If the problem persists, use synthetic data for demonstration.")
