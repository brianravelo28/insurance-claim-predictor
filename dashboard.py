"""
Florida Hurricane Claim Severity Predictor - Interactive Dashboard
Comprehensive analytics platform for claim analysis and model monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING (with caching)
# ============================================================================
@st.cache_data
def load_data():
    """Load datasets from CSV files"""
    try:
        df_claims = pd.read_csv('insurance_day1_clean.csv')
        df_predictions = pd.read_csv('insurance_day2_predictions.csv')

        # Merge datasets
        df = df_claims.merge(
            df_predictions[['claimNumber', 'prediction_log', 'prediction_amount', 'abs_error']],
            on='claimNumber',
            how='left'
        )

        # Convert date columns
        df['dateOfLoss'] = pd.to_datetime(df['dateOfLoss'], errors='coerce')
        df['dateOfClaim'] = pd.to_datetime(df['dateOfClaim'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

if df is None:
    st.error("Could not load data. Please ensure CSV files are in the working directory.")
    st.stop()

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================
with st.sidebar:
    st.title("🎯 Filters")

    # Filter: County
    counties = sorted(df['countyName'].dropna().unique())
    selected_counties = st.multiselect(
        "Select Counties",
        counties,
        default=counties[:3],
        help="Filter claims by Florida counties"
    )

    # Filter: Storm Category
    categories = sorted(df['storm_category'].dropna().unique())
    selected_categories = st.multiselect(
        "Storm Categories",
        categories,
        default=['None', 'Tropical Storm', 'Cat1'],
        help="Filter by hurricane intensity"
    )

    # Filter: Flood Zone
    flood_zones = sorted(df['floodZone'].dropna().unique())
    selected_zones = st.multiselect(
        "Flood Zones",
        flood_zones,
        default=list(flood_zones),
        help="Filter by flood designation"
    )

    # Filter: Building Age Range
    age_range = st.slider(
        "Building Age (years)",
        int(df['building_age_years'].min()),
        int(df['building_age_years'].max()),
        (0, 150),
        help="Filter by building age"
    )

    # Apply filters
    df_filtered = df[
        (df['countyName'].isin(selected_counties)) &
        (df['storm_category'].isin(selected_categories)) &
        (df['floodZone'].isin(selected_zones)) &
        (df['building_age_years'] >= age_range[0]) &
        (df['building_age_years'] <= age_range[1])
    ]

    # Display filter summary
    st.markdown("---")
    st.metric("Filtered Claims", len(df_filtered))
    st.metric("Original Dataset", len(df))

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Explorer",
    "🤖 Model Analysis",
    "📈 Analytics"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.markdown("<div class='header-title'>Dashboard Overview</div>", unsafe_allow_html=True)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Claims",
            f"{len(df_filtered):,}",
            f"{len(df_filtered)/len(df)*100:.1f}% of dataset"
        )

    with col2:
        mean_payout = df_filtered['amountPaid'].mean()
        st.metric(
            "Mean Payout",
            f"${mean_payout:,.0f}",
            f"±${df_filtered['amountPaid'].std():,.0f}"
        )

    with col3:
        median_payout = df_filtered['amountPaid'].median()
        st.metric(
            "Median Payout",
            f"${median_payout:,.0f}",
            delta=None
        )

    with col4:
        model_r2 = 0.0579
        st.metric(
            "Model R²",
            f"{model_r2:.4f}",
            "Test set performance"
        )

    st.markdown("---")

    # Geographic Heatmap
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📍 Claims by County")
        county_counts = df_filtered['countyName'].value_counts().head(10)

        fig_county = go.Figure(data=[
            go.Bar(
                x=county_counts.values,
                y=county_counts.index,
                orientation='h',
                marker=dict(color=county_counts.values, colorscale='Blues')
            )
        ])
        fig_county.update_layout(
            title="Top 10 Counties by Claim Volume",
            xaxis_title="Number of Claims",
            yaxis_title="County",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_county, use_container_width=True)

    with col2:
        st.subheader("📊 Risk Distribution")

        # Risk categories based on prediction
        df_filtered['risk_level'] = pd.cut(
            df_filtered['prediction_amount'],
            bins=[0, 20000, 50000, 100000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        risk_counts = df_filtered['risk_level'].value_counts()

        fig_risk = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker=dict(colors=['green', 'yellow', 'orange', 'red'])
            )
        ])
        fig_risk.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # Top Risk Factors
    st.subheader("🔑 Top Risk Factors")

    feature_importance = {
        'building_age_years': 18.9,
        'latitude': 17.0,
        'longitude': 16.7,
        'claim_lag_days': 16.4,
        'flood_zone_encoded': 10.6,
        'storm_category_encoded': 7.6,
        'historical_storm_freq': 6.6,
        'is_elevated': 2.4,
        'distance_bin': 2.3,
        'occupancy_encoded': 0.9
    }

    fig_features = go.Figure(data=[
        go.Bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            marker=dict(color=list(feature_importance.values()), colorscale='Viridis')
        )
    ])
    fig_features.update_layout(
        title="Feature Importance (Model Ranking)",
        xaxis_title="Importance %",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_features, use_container_width=True)

# ============================================================================
# TAB 2: EXPLORER
# ============================================================================
with tab2:
    st.markdown("<div class='header-title'>Claims Explorer</div>", unsafe_allow_html=True)

    # Interactive scatter plot
    st.subheader("📈 Building Age vs Claim Amount")

    fig_scatter = px.scatter(
        df_filtered,
        x='building_age_years',
        y='amountPaid',
        color='storm_category',
        size='prediction_amount',
        hover_data=['claimNumber', 'countyName', 'latitude', 'longitude'],
        title="Building Age vs Claim Payout (colored by storm category)",
        labels={'building_age_years': 'Building Age (years)', 'amountPaid': 'Claim Amount ($)'},
        height=500
    )
    fig_scatter.update_layout(hovermode='closest')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Data table with sorting
    st.subheader("📋 Claims Data Table")

    display_cols = [
        'claimNumber', 'countyName', 'amountPaid', 'prediction_amount',
        'building_age_years', 'storm_category', 'floodZone', 'latitude', 'longitude'
    ]

    df_display = df_filtered[display_cols].copy()
    df_display['amountPaid'] = df_display['amountPaid'].apply(lambda x: f"${x:,.0f}")
    df_display['prediction_amount'] = df_display['prediction_amount'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

    st.dataframe(
        df_display.head(100),
        use_container_width=True,
        height=500
    )

    # Download data
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_claims.csv",
        mime="text/csv"
    )

# ============================================================================
# TAB 3: MODEL ANALYSIS
# ============================================================================
with tab3:
    st.markdown("<div class='header-title'>Model Analysis & Performance</div>", unsafe_allow_html=True)

    # Model Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Test R² Score", "0.0579", "Variance explained")
    with col2:
        st.metric("Test RMSE", "1.3197", "Log scale error")
    with col3:
        st.metric("Boosting Rounds", "85", "Early stopped")
    with col4:
        st.metric("Training Samples", "80,000", "Training set size")

    st.markdown("---")

    # Predictions vs Actual
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Predicted vs Actual Amount")

        # Only show claims with predictions
        df_pred = df_filtered.dropna(subset=['prediction_amount'])

        fig_pred = px.scatter(
            df_pred,
            x='amountPaid',
            y='prediction_amount',
            title="Model Predictions vs Actual Payouts",
            labels={'amountPaid': 'Actual ($)', 'prediction_amount': 'Predicted ($)'},
            height=400,
            color='storm_category',
            opacity=0.6
        )

        # Add perfect prediction line
        min_val = min(df_pred['amountPaid'].min(), df_pred['prediction_amount'].min())
        max_val = max(df_pred['amountPaid'].max(), df_pred['prediction_amount'].max())

        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))

        fig_pred.update_layout(hovermode='closest')
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        st.subheader("📊 Residuals (Prediction Error)")

        df_pred['residual'] = df_pred['amountPaid'] - df_pred['prediction_amount']

        fig_residual = px.histogram(
            df_pred,
            x='residual',
            nbins=50,
            title="Distribution of Prediction Errors",
            labels={'residual': 'Residual ($)'},
            height=400,
            color_discrete_sequence=['#1f77b4']
        )

        st.plotly_chart(fig_residual, use_container_width=True)

    st.markdown("---")

    # Top Predictions
    st.subheader("🚨 Highest Risk Claims (Top Predictions)")

    df_top = df_filtered.dropna(subset=['prediction_amount']).nlargest(10, 'prediction_amount')

    top_display = df_top[[
        'claimNumber', 'countyName', 'amountPaid', 'prediction_amount',
        'building_age_years', 'storm_category', 'distance_from_track_mi'
    ]].copy()

    top_display['amountPaid'] = top_display['amountPaid'].apply(lambda x: f"${x:,.0f}")
    top_display['prediction_amount'] = top_display['prediction_amount'].apply(lambda x: f"${x:,.0f}")
    top_display['distance_from_track_mi'] = top_display['distance_from_track_mi'].apply(
        lambda x: f"{x:.1f} mi" if pd.notna(x) else "N/A"
    )

    st.dataframe(top_display, use_container_width=True)

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================
with tab4:
    st.markdown("<div class='header-title'>Deep Analytics</div>", unsafe_allow_html=True)

    # Temporal Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Claims by Year")

        claims_by_year = df_filtered.groupby(df_filtered['dateOfLoss'].dt.year).size()

        fig_year = go.Figure(data=[
            go.Bar(x=claims_by_year.index, y=claims_by_year.values, marker_color='steelblue')
        ])
        fig_year.update_layout(
            title="Temporal Distribution of Claims",
            xaxis_title="Year",
            yaxis_title="Number of Claims",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_year, use_container_width=True)

    with col2:
        st.subheader("💰 Payout Distribution")

        fig_payout = px.histogram(
            df_filtered,
            x='amountPaid',
            nbins=50,
            title="Distribution of Claim Amounts",
            labels={'amountPaid': 'Claim Amount ($)'},
            height=400,
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_payout, use_container_width=True)

    st.markdown("---")

    # Feature Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Impact of Building Age")

        age_bins = pd.cut(df_filtered['building_age_years'], bins=5)
        age_analysis = df_filtered.groupby(age_bins)['amountPaid'].agg(['mean', 'count'])

        fig_age = go.Figure(data=[
            go.Bar(
                x=age_analysis.index.astype(str),
                y=age_analysis['mean'],
                marker_color='lightblue',
                text=age_analysis['count'],
                textposition='outside'
            )
        ])
        fig_age.update_layout(
            title="Mean Payout by Building Age",
            xaxis_title="Building Age",
            yaxis_title="Mean Payout ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.subheader("🌊 Flood Zone Impact")

        flood_analysis = df_filtered.groupby('floodZone')['amountPaid'].agg(['mean', 'count']).fillna(0)

        fig_flood = go.Figure(data=[
            go.Bar(
                x=flood_analysis.index,
                y=flood_analysis['mean'],
                marker_color='orange',
                text=flood_analysis['count'],
                textposition='outside'
            )
        ])
        fig_flood.update_layout(
            title="Mean Payout by Flood Zone",
            xaxis_title="Flood Zone",
            yaxis_title="Mean Payout ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_flood, use_container_width=True)

    st.markdown("---")

    # Summary Statistics
    st.subheader("📊 Summary Statistics")

    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Claims',
            'Mean Payout',
            'Median Payout',
            'Std Dev',
            'Min Payout',
            'Max Payout',
            'Storm-Matched Claims',
            'Mean Building Age',
            'Mean Claim Lag (days)'
        ],
        'Value': [
            f"{len(df_filtered):,}",
            f"${df_filtered['amountPaid'].mean():,.0f}",
            f"${df_filtered['amountPaid'].median():,.0f}",
            f"${df_filtered['amountPaid'].std():,.0f}",
            f"${df_filtered['amountPaid'].min():,.0f}",
            f"${df_filtered['amountPaid'].max():,.0f}",
            f"{df_filtered['nearest_storm_name'].notna().sum():,} ({df_filtered['nearest_storm_name'].notna().sum()/len(df_filtered)*100:.1f}%)",
            f"{df_filtered['building_age_years'].mean():.1f}",
            f"{df_filtered['claim_lag_days'].mean():.1f}"
        ]
    })

    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Florida Hurricane Claim Severity Predictor |
    Dataset: 100,000 claims |
    Model: LightGBM (R² = 0.0579) |
    Last Updated: 2026-09-21</p>
    </div>
    """, unsafe_allow_html=True)
