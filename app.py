import asyncio
import os
import warnings
import joblib
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from dotenv import load_dotenv
from db import (
    get_all_data,
    init_database,
    insert_country_data,
    load_countries,
    load_economic_data,
    load_emission_data,
)
from models import CountryData

warnings.filterwarnings("ignore")
load_dotenv()


class DataPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load("co2_emission_model.pkl")
        except FileNotFoundError:
            st.warning("ML model not found. Please train the model first.")

    async def process_csv_data(self, uploaded_file):
        try:
            df = pl.read_csv(
                uploaded_file, null_values=["", " ", "NA", "N/A", "nan", "NaN"]
            )
            df = self.clean_data(df)
            success_count = 0
            error_count = 0
            for row in df.iter_rows(named=True):
                try:
                    validated_data = CountryData(**row)
                    success = insert_country_data(validated_data)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    st.error(f"Validation error for row: {str(e)}")
            return success_count, error_count
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return 0, 0

    def clean_data(self, df):
        threshold = 0.5
        null_counts = df.null_count()
        columns_to_drop = [
            col
            for col, null_count in zip(df.columns, null_counts.row(0))
            if null_count / df.height > threshold
        ]
        if columns_to_drop:
            df = df.drop(columns_to_drop)
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        df = df.with_columns(
            [
                pl.col(col).fill_null(pl.col(col).median())
                for col in numeric_cols
            ]
        )
        key_features = [
            "Total CO2 Emission excluding LUCF (Mt)",
            "GDP PER CAPITA (USD)",
            "Population",
        ]
        for feature in key_features:
            if feature in df.columns:
                q1 = df.select(pl.col(feature).quantile(0.25)).item()
                q3 = df.select(pl.col(feature).quantile(0.75)).item()
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df = df.filter(
                    (pl.col(feature) >= lower) & (pl.col(feature) <= upper)
                )
        return df

    def predict_co2_emissions(self, population, gdp_per_capita, energy_mt):
        if self.model is None:
            return None
        try:
            features = np.array([[population, gdp_per_capita, energy_mt]])
            prediction = self.model.predict(features)[0]
            return max(0, prediction)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None


def create_sidebar():
    st.sidebar.markdown("# 🌍 CO2 Emissions Pipeline")
    tabs = [
        "Dashboard",
        "Data Input",
        "Bulk Upload",
        "Predictions",
        "Analytics",
    ]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    st.sidebar.markdown("### Navigation")
    for tab in tabs:
        if st.sidebar.button(
            f"📊 {tab}"
            if tab == "Dashboard"
            else f"✏️ {tab}"
            if tab == "Data Input"
            else f"📁 {tab}"
            if tab == "Bulk Upload"
            else f"🔮 {tab}"
            if tab == "Predictions"
            else f"📈 {tab}",
            key=f"tab_{tab}",
            use_container_width=True,
            type="primary"
            if st.session_state.active_tab == tab
            else "secondary",
        ):
            st.session_state.active_tab = tab
            st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About This Pipeline**\n\n"
        "This application processes CO2 emissions data with:\n"
        "• Real-time visualization\n"
        "• ML-powered predictions\n"
        "• Advanced analytics\n"
        "• Data validation & cleaning"
    )
    return st.session_state.active_tab


def create_data_input_form():
    st.header("✏️ Manual Data Entry")
    st.markdown("Add individual country emission records to the database")
    with st.form("data_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🌍 Location Info**")
            country = st.text_input("Country", value="Kenya")
            sub_region = st.text_input("Sub-Region", value="Eastern Africa")
            code = st.text_input("Country Code", value="KEN")
            year = st.number_input(
                "Year", min_value=1990, max_value=2030, value=2020
            )
        with col2:
            st.markdown("**💰 Economic Data**")
            population = st.number_input(
                "Population", min_value=1, value=50000000
            )
            gdp_per_capita = st.number_input(
                "GDP Per Capita (USD)", min_value=0.0, value=2000.0
            )
            gdp_per_capita_ppp = st.number_input(
                "GDP Per Capita PPP (USD)", min_value=0.0, value=5000.0
            )
            area_km2 = st.number_input(
                "Area (Km²)", min_value=0.0, value=580000.0
            )
        with col3:
            st.markdown("**🏭 Emission Data**")
            total_co2_excluding_lucf = st.number_input(
                "Total CO2 Excluding LUCF (Mt)", min_value=0.0, value=20.0
            )
            energy_mt = st.number_input(
                "Energy (Mt)", min_value=0.0, value=15.0
            )
            transportation_mt = st.number_input(
                "Transportation (Mt)", min_value=0.0, value=5.0
            )
            electricity_heat = st.number_input(
                "Electricity/Heat (Mt)", min_value=0.0, value=8.0
            )
        submitted = st.form_submit_button(
            "💾 Submit Data", type="primary", use_container_width=True
        )
        if submitted:
            try:
                data = CountryData(
                    country=country,
                    sub_region=sub_region,
                    code=code,
                    year=year,
                    population=population,
                    gdp_per_capita=gdp_per_capita,
                    gdp_per_capita_ppp=gdp_per_capita_ppp,
                    area_km2=area_km2,
                    total_co2_excluding_lucf=total_co2_excluding_lucf,
                    energy_mt=energy_mt,
                    transportation_mt=transportation_mt,
                    electricity_heat=electricity_heat,
                )
                success = insert_country_data(data)
                if success:
                    st.success("✅ Data inserted successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to insert data")
            except Exception as e:
                st.error(f"❌ Validation error: {str(e)}")


def create_dashboard(df):
    st.header("📊 CO2 Emissions Dashboard")
    st.markdown("Real-time insights into global CO2 emissions data")
    if df.empty:
        st.warning(
            "⚠️ No data available. Please add some data using the 'Data Input' or 'Bulk Upload' tabs."
        )
        st.info(
            "💡 **Getting Started:** Use the sidebar navigation to add your first dataset!"
        )
        return
    st.subheader("🔢 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_countries = df["country"].nunique()
        st.metric("Countries", total_countries)
        st.caption("Total unique countries in dataset")
    with col2:
        avg_emissions = df["total_co2_excluding_lucf"].mean()
        st.metric("Avg CO2 Emissions (Mt)", f"{avg_emissions:.2f}")
        st.caption("Mean CO2 emissions per country")
    with col3:
        latest_year = df["year"].max()
        st.metric("Latest Year", latest_year)
        st.caption("Most recent data available")
    with col4:
        total_records = len(df)
        st.metric("Total Records", total_records)
        st.caption("Complete data entries")
    st.subheader("📈 Emissions Trends Over Time")
    fig_timeline = px.line(
        df.groupby(["country", "year"])["total_co2_excluding_lucf"]
        .mean()
        .reset_index(),
        x="year",
        y="total_co2_excluding_lucf",
        color="country",
        title="CO2 Emissions Trajectory by Country",
        labels={
            "total_co2_excluding_lucf": "CO2 Emissions (Mt)",
            "year": "Year",
            "country": "Country",
        },
    )
    fig_timeline.update_layout(height=500)
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.caption(
        "📝 **Analysis:** This chart shows how CO2 emissions have changed over time for each country. Rising trends indicate increasing emissions, while falling trends suggest successful reduction efforts."
    )
    st.subheader("🌏 Regional Emissions Comparison")
    regional_data = (
        df.groupby("sub_region")["total_co2_excluding_lucf"]
        .mean()
        .reset_index()
    )
    fig_regional = px.bar(
        regional_data.sort_values("total_co2_excluding_lucf", ascending=False),
        x="sub_region",
        y="total_co2_excluding_lucf",
        title="Average CO2 Emissions by Region",
        labels={
            "total_co2_excluding_lucf": "Average CO2 Emissions (Mt)",
            "sub_region": "Region",
        },
    )
    fig_regional.update_xaxes(tickangle=45)
    fig_regional.update_layout(height=400)
    st.plotly_chart(fig_regional, use_container_width=True)
    st.caption(
        "📝 **Regional Insights:** Compare emission levels across different world regions. This helps identify which regions contribute most to global emissions."
    )
    st.subheader("🏆 Top CO2 Emitters")
    latest_data = df[df["year"] == df["year"].max()]
    if not latest_data.empty:
        top_emitters = latest_data.sort_values(
            "total_co2_excluding_lucf", ascending=False
        ).head(10)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_bar = px.bar(
                top_emitters,
                x="country",
                y="total_co2_excluding_lucf",
                title=f"Top 10 CO2 Emitters in {latest_data['year'].iloc[0]}",
                labels={
                    "total_co2_excluding_lucf": "CO2 Emissions (Mt)",
                    "country": "Country",
                },
                color="total_co2_excluding_lucf",
                color_continuous_scale="Reds",
            )
            fig_bar.update_xaxes(tickangle=45)
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.markdown("**🎯 Top 10 Emitters**")
            for i, (_, row) in enumerate(top_emitters.iterrows(), 1):
                st.markdown(
                    f"{i}. **{row['country']}**: {row['total_co2_excluding_lucf']:.1f} Mt"
                )
    st.caption(
        "📝 **Impact Analysis:** These countries have the highest absolute CO2 emissions and represent key targets for global emission reduction strategies."
    )
    st.subheader("💰 Economic Development vs Emissions")
    fig_scatter = px.scatter(
        df,
        x="gdp_per_capita",
        y="total_co2_excluding_lucf",
        color="sub_region",
        size="population",
        hover_data=["country", "year"],
        title="GDP per Capita vs CO2 Emissions",
        labels={
            "gdp_per_capita": "GDP per Capita (USD)",
            "total_co2_excluding_lucf": "CO2 Emissions (Mt)",
            "sub_region": "Region",
        },
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "📝 **Economic Correlation:** This scatter plot reveals the relationship between economic development (GDP per capita) and CO2 emissions. Bubble size represents population. Look for patterns between wealth and emissions."
    )
    st.subheader("🔗 Data Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Key Variables",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption(
            "📝 **Correlation Insights:** Red indicates positive correlation, blue indicates negative correlation. Strong correlations (close to +1 or -1) suggest important relationships between variables."
        )


def create_prediction_interface(pipeline):
    st.header("🔮 CO2 Emissions Prediction")
    st.markdown(
        "Use machine learning to predict CO2 emissions based on country characteristics"
    )
    if pipeline.model is None:
        st.error("❌ **ML Model Not Available**")
        st.info(
            "Please ensure the trained model file 'co2_emission_model.pkl' is available in the application directory."
        )
        return
    st.success("✅ **ML Model Ready**")
    st.info(
        "💡 **How it works:** Enter country parameters below to get AI-powered CO2 emission predictions. "
        "The model uses population, GDP per capita, and energy consumption to estimate emissions."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Input Parameters**")
        pred_population = st.number_input(
            "Population",
            min_value=1000,
            max_value=2000000000,
            value=50000000,
            key="pred_pop",
            help="Total population of the country",
        )
        pred_gdp = st.number_input(
            "GDP Per Capita (USD)",
            min_value=0.0,
            max_value=100000.0,
            value=2000.0,
            key="pred_gdp",
            help="Economic output per person",
        )
        pred_energy = st.number_input(
            "Energy Consumption (Mt)",
            min_value=0.0,
            max_value=1000.0,
            value=15.0,
            key="pred_energy",
            help="Total energy consumption in megatons",
        )
    with col2:
        st.markdown("**🎯 Prediction Results**")
        if st.button(
            "🚀 Predict CO2 Emissions", type="primary", use_container_width=True
        ):
            with st.spinner("🤖 Running ML prediction..."):
                prediction = pipeline.predict_co2_emissions(
                    pred_population, pred_gdp, pred_energy
                )
                if prediction is not None:
                    st.success(
                        f"**Predicted CO2 Emissions: {prediction:.2f} Mt**"
                    )
                    st.markdown("**📋 Input Summary:**")
                    st.write(f"• Population: {pred_population:,}")
                    st.write(f"• GDP Per Capita: ${pred_gdp:,.2f}")
                    st.write(f"• Energy Consumption: {pred_energy:.2f} Mt")
                    st.markdown("**💡 Insights:**")
                    if prediction > 100:
                        st.warning(
                            "⚠️ High emission prediction - consider green energy initiatives"
                        )
                    elif prediction > 50:
                        st.info(
                            "📊 Moderate emission levels - room for improvement"
                        )
                    else:
                        st.success(
                            "✅ Relatively low emissions - good environmental performance"
                        )
                else:
                    st.error(
                        "❌ Prediction failed. Please check model availability."
                    )
    st.markdown("---")
    st.caption(
        "🤖 **Model Info:** Predictions are based on historical data patterns and should be used as estimates for planning purposes."
    )


@st.cache_resource
def initialize_app():
    with st.spinner("🔧 Initializing database and pipeline..."):
        init_database()
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory {data_dir} does not exist")
            st.warning(
                "⚠️ Data directory not found. Please ensure CSV files are in the 'data' folder."
            )
            pipeline = DataPipeline()
            return pipeline
        required_files = ["countries.csv", "economic_data.csv", "emissions.csv"]
        missing_files = []
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        if missing_files:
            print(f"Warning: Missing CSV files: {missing_files}")
            st.warning(f"⚠️ Missing CSV files: {', '.join(missing_files)}")
            pipeline = DataPipeline()
            return pipeline
        try:
            print("Loading countries...")
            country_map = load_countries(
                os.path.join(data_dir, "countries.csv")
            )
            if not country_map:
                print("Failed to load countries")
                st.error("❌ Failed to load countries data")
                pipeline = DataPipeline()
                return pipeline
            print("Loading economic data...")
            load_economic_data(
                os.path.join(data_dir, "economic_data.csv"), country_map
            )
            print("Loading emission data...")
            load_emission_data(
                os.path.join(data_dir, "emissions.csv"), country_map
            )
            print("Data loading complete with proper relationships.")
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            st.error(f"❌ Error loading data: {str(e)}")
    pipeline = DataPipeline()
    return pipeline


async def main():
    st.set_page_config(
        page_title="CO2 Emissions Pipeline",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stButton>button {
            width: 100%;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    pipeline = initialize_app()
    page = create_sidebar()
    df = get_all_data()
    if page == "Dashboard":
        create_dashboard(df)
    elif page == "Data Input":
        create_data_input_form()
    elif page == "Bulk Upload":
        st.header("📁 Bulk Data Upload")
        st.markdown(
            "Upload CSV files containing multiple country emission records"
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with CO2 emissions data following the required format",
        )
        if uploaded_file is not None:
            st.info(
                "📋 **File uploaded successfully!** Click 'Process File' to validate and import the data."
            )
            if st.button(
                "⚙️ Process File", type="primary", use_container_width=True
            ):
                with st.spinner("🔄 Processing and validating data..."):
                    (
                        success_count,
                        error_count,
                    ) = await pipeline.process_csv_data(uploaded_file)
                    if success_count > 0:
                        st.success(
                            f"✅ **Success!** Processed {success_count} records successfully"
                        )
                        if error_count > 0:
                            st.warning(
                                f"⚠️ {error_count} records had validation errors"
                            )
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(
                            f"❌ **Failed to process data.** {error_count} validation errors occurred"
                        )
                        st.info(
                            "💡 Please check your CSV format and data types"
                        )
    elif page == "Predictions":
        create_prediction_interface(pipeline)
    elif page == "Analytics":
        st.header("📈 Advanced Analytics")
        st.markdown("Deep dive into emission patterns and statistical insights")
        if not df.empty:
            tab1, tab2, tab3 = st.tabs(
                ["📊 Statistical Summary", "🔍 Data Explorer", "📋 Raw Data"]
            )
            with tab1:
                st.subheader("Statistical Overview")
                st.dataframe(df.describe(), use_container_width=True)
                st.subheader("Key Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🌍 Geographic Distribution**")
                    region_counts = df["sub_region"].value_counts()
                    st.bar_chart(region_counts)
                with col2:
                    st.markdown("**📅 Temporal Coverage**")
                    year_counts = df["year"].value_counts().sort_index()
                    st.line_chart(year_counts)
            with tab2:
                st.subheader("Interactive Data Explorer")
                col1, col2, col3 = st.columns(3)
                with col1:
                    selected_countries = st.multiselect(
                        "Select Countries",
                        options=df["country"].unique(),
                        default=df["country"].unique()[:5],
                    )
                with col2:
                    year_range = st.slider(
                        "Year Range",
                        min_value=int(df["year"].min()),
                        max_value=int(df["year"].max()),
                        value=(int(df["year"].min()), int(df["year"].max())),
                    )
                with col3:
                    selected_regions = st.multiselect(
                        "Select Regions",
                        options=df["sub_region"].unique(),
                        default=df["sub_region"].unique(),
                    )
                filtered_df = df[
                    (df["country"].isin(selected_countries))
                    & (df["year"] >= year_range[0])
                    & (df["year"] <= year_range[1])
                    & (df["sub_region"].isin(selected_regions))
                ]
                if not filtered_df.empty:
                    st.dataframe(filtered_df, use_container_width=True)
                    st.caption(
                        f"Showing {len(filtered_df)} records based on your filters"
                    )
                else:
                    st.warning("No data matches your current filters")
            with tab3:
                st.subheader("Complete Dataset")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Total records: {len(df)}")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Dataset",
                    data=csv,
                    file_name="co2_emissions_data.csv",
                    mime="text/csv",
                )
        else:
            st.warning(
                "⚠️ No data available for analytics. Please add data using other tabs."
            )
            st.info(
                "💡 Use 'Data Input' for manual entry or 'Bulk Upload' for CSV files"
            )


if __name__ == "__main__":
    asyncio.run(main())