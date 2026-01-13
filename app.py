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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .alert-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .alert-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .alert-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
    }
    .link-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .link-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
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
    # Header with improved design
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; font-size: 2.8rem; margin: 0; font-weight: 800;">☀️ Solar PV Fault Detection System</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;">AI-Powered Predictive Maintenance & Performance Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with professional styling
    with st.sidebar:
        # Logo section
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">⚡</div>
            <h2 style="color: #2d3748; font-weight: 700; margin: 0.5rem 0;">PVGuard AI</h2>
            <p style="color: #718096; font-size: 0.9rem; margin: 0;">Intelligent Solar Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)
        
        analysis_days = st.slider("Analysis Period (days)", 7, 90, 30, 
                                 help="Number of days to analyze")
        contamination = st.slider("Anomaly Detection Sensitivity", 0.01, 0.3, 0.1, 0.01,
                                 help="Higher values detect more anomalies but may increase false positives")
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">System Parameters</div>', unsafe_allow_html=True)
        
        system_capacity = st.number_input("System Capacity (kW)", 10, 1000, 100, step=10,
                                         help="Total installed capacity of the PV system")
        location = st.selectbox("Location", ["Desert", "Temperate", "Coastal", "Urban"],
                               help="Geographical location affecting performance benchmarks")
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">Data Source</div>', unsafe_allow_html=True)
        
        data_source = st.radio("Choose data source:", 
                              ["Use Synthetic Data", "Upload CSV"],
                              horizontal=True)
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload PV data CSV", 
                                            type=['csv'],
                                            help="Upload your solar PV operational data in CSV format")
        
        # Professional Developer Profile Section
        st.markdown("---")
        st.markdown('<div class="sidebar-title">🏆 Developed By</div>', unsafe_allow_html=True)
        
        # Profile Card
        st.markdown("""
        <div class="profile-card">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 2.5rem;">👨‍💻</div>
                <div>
                    <h3 style="margin: 0; color: white;">Areeb Rizwan</h3>
                    <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Renewable Energy Data Scientist</p>
                </div>
            </div>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Specializing in AI/ML solutions for clean energy optimization and predictive maintenance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact Links
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <h4 style="color: #2d3748; margin-bottom: 0.8rem;">🌐 Connect With Me</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Portfolio Link
        st.markdown("""
        <a href="http://www.areebrizwan.com" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 0.8rem;">
                        <div style="font-size: 1.5rem;">🌍</div>
                        <div>
                            <div style="font-weight: 600; color: #2d3748;">Portfolio Website</div>
                            <div style="font-size: 0.8rem; color: #718096;">www.areebrizwan.com</div>
                        </div>
                    </div>
                    <div style="color: #667eea; font-size: 1.2rem;">→</div>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
        # LinkedIn Link
        st.markdown("""
        <a href="https://www.linkedin.com/in/areebrizwan" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 0.8rem;">
                        <div style="font-size: 1.5rem;">💼</div>
                        <div>
                            <div style="font-weight: 600; color: #2d3748;">LinkedIn Profile</div>
                            <div style="font-size: 0.8rem; color: #718096;">Professional Network</div>
                        </div>
                    </div>
                    <div style="color: #667eea; font-size: 1.2rem;">→</div>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
        # GitHub Link
        st.markdown("""
        <a href="https://github.com/areebrizwan" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 0.8rem;">
                        <div style="font-size: 1.5rem;">💻</div>
                        <div>
                            <div style="font-weight: 600; color: #2d3748;">GitHub</div>
                            <div style="font-size: 0.8rem; color: #718096;">Open Source Projects</div>
                        </div>
                    </div>
                    <div style="color: #667eea; font-size: 1.2rem;">→</div>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
        # Copyright Notice
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
            <p style="text-align: center; color: #718096; font-size: 0.8rem; margin: 0;">
                © 2024 Areeb Rizwan<br>
                All rights reserved
            </p>
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
            st.success(f"✅ Synthetic data generated for {analysis_days} days")
            
        elif uploaded_file is not None:
            with st.spinner("Loading uploaded data..."):
                df = pd.read_csv(uploaded_file)
            st.success("✅ Data loaded successfully!")
            
            # Check for required columns
            required_cols = ['timestamp', 'voltage', 'current', 'power', 
                            'temperature', 'irradiance']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns in uploaded data: {missing_cols}")
                st.info("ℹ️ The system will try to work with available data. Synthetic data may give better results.")
        else:
            st.info("📊 Please upload a CSV file or use synthetic data to begin analysis.")
            
            # Show sample data structure
            with st.expander("📋 Expected Data Format", expanded=True):
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
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("ℹ️ Using synthetic data instead.")
        df = generate_synthetic_data(days=analysis_days)
    
    # Add system capacity to data
    df['system_capacity'] = system_capacity
    
    # Process data with progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🧹 Cleaning and validating data..."):
        status_text.text("Step 1/5: Data Cleaning")
        df = analyzer.clean_data(df)
        progress_bar.progress(20)
    
    with st.spinner("📊 Calculating performance metrics..."):
        status_text.text("Step 2/5: Performance Metrics")
        df = analyzer.compute_performance_metrics(df)
        progress_bar.progress(40)
    
    with st.spinner("🔧 Engineering features for analysis..."):
        status_text.text("Step 3/5: Feature Engineering")
        df = analyzer.engineer_features(df)
        progress_bar.progress(60)
    
    with st.spinner("🤖 Running anomaly detection..."):
        status_text.text("Step 4/5: Anomaly Detection")
        df = analyzer.detect_anomalies(df)
        progress_bar.progress(80)
    
    with st.spinner("📈 Calculating trends and severity..."):
        status_text.text("Step 5/5: Trend Analysis")
        daily_perf, monthly_perf = analyzer.calculate_degradation(df)
        fault_severity = analyzer.calculate_fault_severity(df)
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
    
    # Dashboard metrics with enhanced styling
    st.markdown("### 📊 System Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pr_mean = df['performance_ratio'].mean() if 'performance_ratio' in df.columns else 0
        delta_pr = f"{((pr_mean - 0.8)/0.8*100):+.1f}%" if pr_mean > 0 else "N/A"
        st.metric("Performance Ratio", 
                 f"{pr_mean:.2%}",
                 delta_pr,
                 help="IEC 61724 Performance Ratio - Ratio of actual to expected output")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        cf_mean = df['capacity_factor'].mean() if 'capacity_factor' in df.columns else 0
        delta_cf = f"{((cf_mean - 0.2)/0.2*100):+.1f}%" if cf_mean > 0 else "N/A"
        st.metric("Capacity Factor", 
                 f"{cf_mean:.2%}",
                 delta_cf,
                 help="Ratio of actual energy output to maximum possible output")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        anomalies = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
        anomaly_rate = f"{anomalies/len(df)*100:.1f}%" if len(df) > 0 else "0%"
        st.metric("Anomalies Detected", 
                 f"{int(anomalies)}",
                 anomaly_rate,
                 help="Number of abnormal operating conditions detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        severity_class = "High" if fault_severity > 7 else "Medium" if fault_severity > 4 else "Low"
        severity_color = "🔴" if fault_severity > 7 else "🟡" if fault_severity > 4 else "🟢"
        st.metric("Fault Severity Index", 
                 f"{fault_severity:.2f}/10",
                 f"{severity_color} {severity_class}",
                 help="Overall severity score based on anomaly impact and frequency")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs with enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Overview", 
        "🚨 Fault Analysis", 
        "📈 Degradation Trends",
        "🔧 Maintenance Dashboard"
    ])
    
    with tab1:
        st.markdown("### 📈 Performance Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            # Power vs Irradiance
            if all(col in df.columns for col in ['irradiance', 'power', 'is_anomaly']):
                fig = px.scatter(df, x='irradiance', y='power', 
                                color=df['is_anomaly'].astype(str),
                                title='Power vs Irradiance',
                                labels={'color': 'Anomaly Status'},
                                color_discrete_map={'0': '#38ef7d', '1': '#ff416c'},
                                opacity=0.7,
                                trendline="lowess",
                                trendline_color_override="#667eea")
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient data for Power vs Irradiance plot")
            
            # Performance Ratio Distribution
            if 'performance_ratio' in df.columns:
                fig = px.histogram(df, x='performance_ratio', 
                                  title='Performance Ratio Distribution',
                                  nbins=30,
                                  color_discrete_sequence=['#667eea'],
                                  opacity=0.8)
                fig.add_vline(x=df['performance_ratio'].mean(), 
                             line_dash="dash", 
                             line_color="#ff416c",
                             annotation_text=f"Mean: {df['performance_ratio'].mean():.2%}",
                             annotation_position="top right")
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time series of key metrics
            if 'timestamp' in df.columns and 'power' in df.columns:
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=('Power Generation (kW)', 'Performance Ratio'),
                                   vertical_spacing=0.12)
                
                # Power plot
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df['power'], 
                              mode='lines',
                              name='Power',
                              line=dict(color='#667eea', width=2),
                              fill='tozeroy',
                              fillcolor='rgba(102, 126, 234, 0.1)'),
                    row=1, col=1
                )
                
                # Anomaly markers for power
                anomalies_df = df[df['is_anomaly'] == 1]
                if not anomalies_df.empty and 'timestamp' in anomalies_df.columns:
                    fig.add_trace(
                        go.Scatter(x=anomalies_df['timestamp'], 
                                  y=anomalies_df['power'],
                                  mode='markers',
                                  name='Anomaly',
                                  marker=dict(color='#ff416c', size=8, symbol='x'),
                                  hovertext=anomalies_df['fault_type'] if 'fault_type' in anomalies_df.columns else 'Anomaly'),
                        row=1, col=1
                    )
                
                # Performance Ratio plot
                if 'performance_ratio' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['performance_ratio'], 
                                  mode='lines',
                                  name='Performance Ratio',
                                  line=dict(color='#11998e', width=2)),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748'),
                    hovermode='x unified'
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🚨 Fault Detection Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Fault type distribution
            if 'fault_type' in df.columns and 'is_anomaly' in df.columns:
                fault_counts = df[df['is_anomaly'] == 1]['fault_type'].value_counts()
                if len(fault_counts) > 0:
                    colors = ['#ff416c', '#ff4b2b', '#f7971e', '#ffd200', '#11998e', '#38ef7d']
                    fig = px.pie(values=fault_counts.values, 
                                names=fault_counts.index,
                                title='Fault Type Distribution',
                                hole=0.4,
                                color_discrete_sequence=colors)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#2d3748'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("✅ No anomalies detected in the data")
            else:
                st.warning("⚠️ Fault analysis data not available")
            
            # Anomaly scores distribution
            if 'anomaly_score' in df.columns and 'is_anomaly' in df.columns:
                anomaly_data = df[df['is_anomaly'] == 1]
                if len(anomaly_data) > 0:
                    fig = px.histogram(anomaly_data, 
                                      x='anomaly_score',
                                      title='Anomaly Scores Distribution',
                                      nbins=20,
                                      color_discrete_sequence=['#ff416c'])
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#2d3748')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fault details table
            if all(col in df.columns for col in ['is_anomaly', 'fault_type', 'anomaly_score']):
                anomalies_df = df[df['is_anomaly'] == 1].copy()
                if not anomalies_df.empty:
                    # Fault timeline heatmap
                    if 'timestamp' in anomalies_df.columns:
                        anomalies_df['hour'] = pd.to_datetime(anomalies_df['timestamp']).dt.hour
                        anomalies_df['date'] = pd.to_datetime(anomalies_df['timestamp']).dt.date
                        
                        # Create pivot table for heatmap
                        heatmap_data = anomalies_df.groupby(['date', 'hour'])['anomaly_score'].mean().unstack().fillna(0)
                        
                        fig = px.imshow(heatmap_data.values,
                                       labels=dict(x="Hour of Day", y="Date", color="Anomaly Score"),
                                       x=list(range(24)),
                                       y=heatmap_data.index.astype(str),
                                       title='Fault Occurrence Timeline',
                                       color_continuous_scale='Reds',
                                       aspect='auto')
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#2d3748')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Fault details table with enhanced styling
                    st.markdown("### 📋 Detected Faults Details")
                    
                    # Style the dataframe
                    display_cols = ['timestamp', 'fault_type', 'anomaly_score']
                    if 'power' in anomalies_df.columns:
                        display_cols.append('power')
                    if 'performance_ratio' in anomalies_df.columns:
                        display_cols.append('performance_ratio')
                    
                    available_cols = [col for col in display_cols if col in anomalies_df.columns]
                    fault_details = anomalies_df[available_cols].copy()
                    
                    if 'timestamp' in fault_details.columns:
                        fault_details['timestamp'] = pd.to_datetime(fault_details['timestamp'])
                        fault_details['timestamp'] = fault_details['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    
                    if 'anomaly_score' in fault_details.columns:
                        fault_details = fault_details.sort_values('anomaly_score', ascending=False)
                    
                    # Display with custom styling
                    st.dataframe(
                        fault_details.head(20).style
                        .background_gradient(subset=['anomaly_score'], cmap='Reds')
                        .format({'anomaly_score': '{:.3f}'}),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.success("""
                    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 12px; color: white;">
                        <div style="font-size: 4rem;">✅</div>
                        <h2 style="color: white;">No Anomalies Detected</h2>
                        <p>Your system is operating within normal parameters.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Anomaly detection results not available")
    
    with tab3:
        st.markdown("### 📈 Performance Degradation Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily performance
            if not daily_perf.empty and 'performance_ratio' in daily_perf.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                        y=daily_perf['performance_ratio'],
                                        mode='lines+markers',
                                        name='Performance Ratio',
                                        line=dict(color='#667eea', width=3),
                                        marker=dict(size=6)))
                
                if 'irradiance' in daily_perf.columns:
                    fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                            y=daily_perf['irradiance']/1000,
                                            mode='lines',
                                            name='Irradiance (kW/m²)',
                                            yaxis='y2',
                                            line=dict(color='#f7971e', dash='dash', width=2)))
                
                fig.update_layout(
                    title='Daily Performance Trend',
                    yaxis=dict(title='Performance Ratio'),
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748')
                )
                
                if 'irradiance' in daily_perf.columns:
                    fig.update_layout(
                        yaxis2=dict(title='Irradiance (kW/m²)', 
                                   overlaying='y', 
                                   side='right',
                                   color='#f7971e')
                    )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Daily performance data not available")
        
        with col2:
            # Monthly degradation
            if not monthly_perf.empty and 'performance_ratio' in monthly_perf.columns:
                fig = make_subplots(rows=2, cols=1,
                                   subplot_titles=('Monthly Performance Ratio', 
                                                  'Monthly Degradation Rate (%)'),
                                   vertical_spacing=0.15)
                
                # Performance ratio bar chart
                fig.add_trace(
                    go.Bar(x=monthly_perf['month'].astype(str), 
                          y=monthly_perf['performance_ratio'],
                          name='Performance Ratio',
                          marker_color='#667eea',
                          text=monthly_perf['performance_ratio'].round(2),
                          textposition='outside'),
                    row=1, col=1
                )
                
                # Degradation rate line chart
                if 'degradation_rate' in monthly_perf.columns:
                    fig.add_trace(
                        go.Scatter(x=monthly_perf['month'].astype(str), 
                                  y=monthly_perf['degradation_rate'],
                                  mode='lines+markers',
                                  name='Degradation Rate',
                                  line=dict(color='#ff416c', width=3),
                                  marker=dict(size=8, symbol='diamond')),
                        row=2, col=1
                    )
                    
                    # Add zero line for reference
                    fig.add_hline(y=0, line_dash="dot", line_color="#718096", row=2, col=1)
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2d3748')
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f2f6')
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔧 Maintenance Priority Dashboard")
        
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
                        return ('High', '🔴', '#ff416c')
                    elif score > 4:
                        return ('Medium', '🟡', '#f7971e')
                    else:
                        return ('Low', '🟢', '#38ef7d')
                
                fault_summary[['Priority', 'Priority Icon', 'Priority Color']] = fault_summary.apply(
                    assign_priority, axis=1, result_type='expand'
                )
                
                fault_summary = fault_summary.sort_values('Avg Severity', ascending=False)
                
                # Display maintenance table with enhanced styling
                st.markdown("### 📋 Maintenance Priority Summary")
                
                # Create a styled dataframe
                styled_df = fault_summary.style
                
                # Apply background gradient for severity
                styled_df = styled_df.background_gradient(subset=['Avg Severity'], cmap='Reds')
                
                # Apply color coding for Priority column
                def color_priority(val):
                    if val == 'High':
                        return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
                    elif val == 'Medium':
                        return 'background-color: #ffe6cc; color: #e68a00; font-weight: bold'
                    else:
                        return 'background-color: #ccffcc; color: #006600; font-weight: bold'
                
                styled_df = styled_df.applymap(color_priority, subset=['Priority'])
                
                # Display the styled dataframe
                st.dataframe(styled_df, use_container_width=True)
                
                # Maintenance recommendations with enhanced styling
                st.markdown("### 🛠️ Recommended Actions")
                
                recommendations = {
                    'Soiling': [
                        "Schedule panel cleaning within 1 week",
                        "Inspect automatic cleaning system",
                        "Review weather patterns and dust accumulation"
                    ],
                    'Shading': [
                        "Trim surrounding vegetation immediately",
                        "Review panel positioning and orientation",
                        "Consider micro-inverters for shaded panels"
                    ],
                    'Inverter Degradation': [
                        "Schedule inverter inspection within 2 weeks",
                        "Check warranty status and replacement timeline",
                        "Monitor efficiency metrics daily"
                    ],
                    'Electrical Fault': [
                        "Immediate electrical safety inspection required",
                        "Check all connections and wiring",
                        "Review grounding and surge protection"
                    ],
                    'Overheating': [
                        "Improve ventilation around equipment",
                        "Check cooling system functionality",
                        "Review installation location for thermal management"
                    ],
                    'Unknown Anomaly': [
                        "Further investigation required",
                        "Check all system components systematically",
                        "Review historical data patterns"
                    ]
                }
                
                for fault_type in fault_summary['Fault Type'].unique():
                    if fault_type in recommendations:
                        priority_row = fault_summary[fault_summary['Fault Type'] == fault_type].iloc[0]
                        priority = priority_row['Priority']
                        priority_color = priority_row['Priority Color']
                        
                        # Create a colored card for each fault type
                        st.markdown(f"""
                        <div style="border-left: 4px solid {priority_color}; padding-left: 1rem; margin: 1rem 0; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h4 style="margin: 0; color: #2d3748;">{fault_type}</h4>
                                <span style="background: {priority_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {priority} Priority
                                </span>
                            </div>
                            <ul style="margin: 0; padding-left: 1.2rem; color: #4a5568;">
                                {"".join([f"<li>{item}</li>" for item in recommendations[fault_type]])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("""
                <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 12px; color: white; margin: 2rem 0;">
                    <div style="font-size: 4rem;">✅</div>
                    <h2 style="color: white;">No Maintenance Required</h2>
                    <p>All systems are operating optimally. Continue with regular preventive maintenance schedule.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance loss estimation
        if 'power_deviation' in df.columns and 'is_anomaly' in df.columns:
            total_power_loss = df[df['is_anomaly'] == 1]['power_deviation'].sum() * 100
            if total_power_loss != 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Estimated Performance Loss", 
                             f"{abs(total_power_loss):.1f}%",
                             f"Equivalent to ${abs(total_power_loss/100 * system_capacity * 0.12 * 24 * 30):.0f}/month",
                             delta_color="inverse",
                             help="Estimated financial impact based on average electricity prices")
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        <div style="padding: 2rem; background: #ffcccc; border-radius: 10px; border-left: 4px solid #cc0000;">
            <h3 style="color: #cc0000; margin: 0;">❌ Application Error</h3>
            <p style="color: #990000; margin: 0.5rem 0;">{str(e)}</p>
            <p style="color: #666; margin: 0;">Please check your data and try again. If the problem persists, use synthetic data for demonstration.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("🔄 Try using synthetic data or check your CSV file format.")
