import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.pca = PCA(n_components=2)
        
    def compute_performance_metrics(self, df):
        """Calculate IEC 61724 performance metrics"""
        
        # Performance Ratio (PR)
        expected_power = df['irradiance'] * df['system_capacity'] * 0.2  # 20% efficiency assumption
        df['performance_ratio'] = (df['power'] / expected_power).clip(upper=1)
        
        # Capacity Factor (CF)
        df['capacity_factor'] = df['power'] / df['system_capacity']
        
        # Normalized power
        df['normalized_power'] = df['power'] / df['irradiance'].replace(0, 1)
        
        # Temperature corrected output
        temp_coeff = -0.004  # Typical temperature coefficient for silicon PV
        ref_temp = 25
        df['temp_corrected_power'] = df['power'] * (1 + temp_coeff * (df['temperature'] - ref_temp))
        
        return df
    
    def engineer_features(self, df):
        """Create engineered features for anomaly detection"""
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        
        # Operational features
        df['current_voltage_ratio'] = df['current'] / (df['voltage'] + 0.001)
        df['power_voltage_ratio'] = df['power'] / (df['voltage'] + 0.001)
        
        # Deviation from expected
        expected_power = df['irradiance'] * df['system_capacity'] * 0.2
        df['power_deviation'] = (df['power'] - expected_power) / expected_power
        
        # Rolling statistics
        df['power_rolling_mean'] = df['power'].rolling(window=24, min_periods=1).mean()
        df['power_rolling_std'] = df['power'].rolling(window=24, min_periods=1).std()
        df['power_zscore'] = (df['power'] - df['power_rolling_mean']) / (df['power_rolling_std'] + 0.001)
        
        return df
    
    def detect_anomalies(self, df):
        """Train Isolation Forest and detect anomalies"""
        
        # Features for anomaly detection
        feature_columns = [
            'normalized_power', 'temp_corrected_power', 
            'performance_ratio', 'capacity_factor',
            'current_voltage_ratio', 'power_deviation',
            'power_zscore'
        ]
        
        # Prepare data
        X = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Predict anomalies
        df['anomaly_score'] = self.model.decision_function(X_scaled)
        df['is_anomaly'] = self.model.predict(X_scaled)
        df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})  # Convert to 0=normal, 1=anomaly
        
        # Fault classification
        df['fault_type'] = self.classify_faults(df)
        
        return df
    
    def classify_faults(self, df):
        """Classify detected anomalies into fault types"""
        
        fault_types = []
        
        for idx, row in df.iterrows():
            if row['is_anomaly'] == 0:
                fault_types.append('Normal')
                continue
                
            # Soiling detection: Low normalized power despite good irradiance
            if (row['normalized_power'] < 0.6 and 
                row['irradiance'] > 500 and
                row['temperature'] < 40):
                fault_types.append('Soiling')
                
            # Shading detection: Sudden drops in current with stable voltage
            elif (row['power_deviation'] < -0.3 and 
                  abs(row['current_voltage_ratio']) > 2):
                fault_types.append('Shading')
                
            # Inverter degradation: Gradual power loss with normal I-V characteristics
            elif (row['performance_ratio'] < 0.7 and 
                  row['capacity_factor'] < 0.5 and
                  0.9 < row['current_voltage_ratio'] < 1.1):
                fault_types.append('Inverter Degradation')
                
            # Temperature issues
            elif row['temperature'] > 60:
                fault_types.append('Overheating')
                
            # Electrical issues
            elif row['voltage'] < 200 or row['current'] < 2:
                fault_types.append('Electrical Fault')
                
            else:
                fault_types.append('Unknown Anomaly')
                
        return fault_types
    
    def calculate_degradation(self, df):
        """Calculate daily and monthly degradation trends"""
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        
        # Daily performance
        daily_perf = df.groupby('date').agg({
            'performance_ratio': 'mean',
            'power': 'sum',
            'irradiance': 'mean'
        }).reset_index()
        
        # Monthly degradation
        monthly_perf = df.groupby('month').agg({
            'performance_ratio': 'mean',
            'capacity_factor': 'mean'
        }).reset_index()
        
        # Calculate degradation rate
        if len(monthly_perf) > 1:
            monthly_perf['degradation_rate'] = monthly_perf['performance_ratio'].pct_change() * 100
        
        return daily_perf, monthly_perf
    
    def calculate_fault_severity(self, df):
        """Calculate fault severity index"""
        
        severity_scores = []
        
        for _, row in df[df['is_anomaly'] == 1].iterrows():
            base_score = abs(row['anomaly_score'])
            
            # Adjust based on fault type
            if row['fault_type'] == 'Inverter Degradation':
                severity = base_score * 1.5
            elif row['fault_type'] == 'Electrical Fault':
                severity = base_score * 1.3
            elif row['fault_type'] == 'Soiling':
                severity = base_score * 1.1
            elif row['fault_type'] == 'Shading':
                severity = base_score * 1.0
            else:
                severity = base_score
            
            # Adjust based on power loss
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
    
    return pd.DataFrame(data)

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
        system_capacity = st.number_input("System Capacity (kW)", 10, 1000, 100)
        location = st.selectbox("Location", ["Desert", "Temperate", "Coastal", "Urban"])
        
        st.markdown("---")
        st.markdown("#### Data Source")
        data_source = st.radio("Choose data source:", 
                              ["Use Synthetic Data", "Upload CSV"])
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload PV data CSV", 
                                            type=['csv'])
            if uploaded_file:
                st.success("File uploaded successfully!")
        
        st.markdown("---")
        st.markdown("### Credits")
        st.markdown("**Made by Areeb Rizwan**")
        st.markdown("[www.areebrizwan.com](http://www.areebrizwan.com)")
        st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/areebrizwan)")
    
    # Initialize analyzer
    analyzer = SolarPVAnalyzer()
    analyzer.model = IsolationForest(contamination=contamination, random_state=42)
    
    # Load data
    if data_source == "Use Synthetic Data":
        df = generate_synthetic_data(days=analysis_days)
        st.info(f"Using synthetic data for {analysis_days} days")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Ensure required columns exist
        required_cols = ['timestamp', 'voltage', 'current', 'power', 
                        'temperature', 'irradiance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
    else:
        st.warning("Please upload a CSV file or use synthetic data")
        return
    
    # Add system capacity to data
    df['system_capacity'] = system_capacity
    
    # Process data
    with st.spinner("Processing PV data and calculating metrics..."):
        df = analyzer.compute_performance_metrics(df)
        df = analyzer.engineer_features(df)
        df = analyzer.detect_anomalies(df)
        daily_perf, monthly_perf = analyzer.calculate_degradation(df)
        fault_severity = analyzer.calculate_fault_severity(df)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Performance Ratio", 
                 f"{df['performance_ratio'].mean():.2%}",
                 f"{((df['performance_ratio'].mean() - 0.8)/0.8*100):+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Capacity Factor", 
                 f"{df['capacity_factor'].mean():.2%}",
                 f"{((df['capacity_factor'].mean() - 0.2)/0.2*100):+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        anomalies = df['is_anomaly'].sum()
        st.metric("Anomalies Detected", 
                 f"{anomalies}",
                 f"{anomalies/len(df)*100:.1f}%")
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
            fig = px.scatter(df, x='irradiance', y='power', 
                            color='is_anomaly',
                            title='Power vs Irradiance',
                            labels={'is_anomaly': 'Anomaly'},
                            color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance Ratio Distribution
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
            fig = make_subplots(rows=2, cols=1, 
                               subplot_titles=('Power Generation', 'Performance Ratio'))
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['power'], 
                          mode='lines', name='Power'),
                row=1, col=1
            )
            
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
            fault_counts = df[df['is_anomaly'] == 1]['fault_type'].value_counts()
            fig = px.pie(values=fault_counts.values, 
                        names=fault_counts.index,
                        title='Fault Type Distribution',
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly scores distribution
            fig = px.histogram(df[df['is_anomaly'] == 1], 
                              x='anomaly_score',
                              title='Anomaly Scores Distribution',
                              nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fault timeline
            anomalies_df = df[df['is_anomaly'] == 1].copy()
            if not anomalies_df.empty:
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
                fault_details = anomalies_df[['timestamp', 'fault_type', 
                                            'anomaly_score', 'power', 
                                            'performance_ratio']].copy()
                fault_details['timestamp'] = pd.to_datetime(fault_details['timestamp'])
                fault_details = fault_details.sort_values('anomaly_score', 
                                                         ascending=False)
                st.dataframe(fault_details.head(20), use_container_width=True)
            else:
                st.success("No anomalies detected!")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily performance
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                    y=daily_perf['performance_ratio'],
                                    mode='lines+markers',
                                    name='Performance Ratio',
                                    line=dict(color='blue')))
            
            fig.add_trace(go.Scatter(x=daily_perf['date'], 
                                    y=daily_perf['irradiance']/1000,
                                    mode='lines',
                                    name='Irradiance (kW/m²)',
                                    yaxis='y2',
                                    line=dict(color='orange', dash='dash')))
            
            fig.update_layout(
                title='Daily Performance Trend',
                yaxis=dict(title='Performance Ratio'),
                yaxis2=dict(title='Irradiance (kW/m²)', 
                           overlaying='y', 
                           side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly degradation
            if len(monthly_perf) > 1 and 'degradation_rate' in monthly_perf.columns:
                fig = make_subplots(rows=2, cols=1,
                                   subplot_titles=('Monthly Performance', 
                                                  'Degradation Rate'))
                
                fig.add_trace(
                    go.Bar(x=monthly_perf['month'].astype(str), 
                          y=monthly_perf['performance_ratio'],
                          name='Performance Ratio'),
                    row=1, col=1
                )
                
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
        fault_summary = df[df['is_anomaly'] == 1].groupby('fault_type').agg({
            'anomaly_score': 'mean',
            'timestamp': 'count',
            'power': 'mean'
        }).reset_index()
        
        fault_summary.columns = ['Fault Type', 'Avg Severity', 'Occurrences', 'Avg Power Loss']
        fault_summary['Power Loss %'] = (fault_summary['Avg Power Loss'] / df['power'].mean()) * 100
        
        # Assign priority scores
        def assign_priority(row):
            score = (row['Avg Severity'] * 0.4 + 
                    (row['Occurrences'] / len(df)) * 100 * 0.3 +
                    row['Power Loss %'] * 0.3)
            
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
            'Overheating': "• Improve ventilation\n• Check cooling systems\n• Review installation location"
        }
        
        for fault_type in fault_summary['Fault Type'].unique():
            if fault_type in recommendations and fault_type != 'Normal':
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
        
        # Performance loss estimation
        total_power_loss = df[df['is_anomaly'] == 1]['power_deviation'].sum() * 100
        st.metric("Estimated Total Performance Loss", 
                 f"{abs(total_power_loss):.1f}%",
                 delta_color="inverse")

if __name__ == "__main__":
    main()
