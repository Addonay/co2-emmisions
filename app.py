import asyncio
import os
import warnings
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from sqlmodel import Field, Session, SQLModel, and_, create_engine, select

warnings.filterwarnings("ignore")
load_dotenv()


# =============================================================================
# PYDANTIC MODELS FOR DATA VALIDATION
# =============================================================================


class CountryData(BaseModel):
    """Pydantic model for input data validation"""

    country: str
    sub_region: str
    code: str
    year: int
    population: int
    gdp_per_capita: float
    gdp_per_capita_ppp: Optional[float] = None
    area_km2: Optional[float] = None
    transportation_mt: Optional[float] = 0.0
    total_co2_including_lucf: Optional[float] = None
    total_co2_excluding_lucf: float
    other_fuel_combustion: Optional[float] = 0.0
    manufacturing_construction: Optional[float] = 0.0
    land_use_change_forestry: Optional[float] = 0.0
    industrial_processes: Optional[float] = 0.0
    energy_mt: Optional[float] = 0.0
    electricity_heat: Optional[float] = 0.0
    bunker_fuels: Optional[float] = 0.0
    building_mt: Optional[float] = 0.0

    @field_validator("year")
    def validate_year(cls, v):
        if v < 1990 or v > 2030:
            raise ValueError("Year must be between 1990 and 2030")
        return v

    @field_validator("population")
    def validate_population(cls, v):
        if v <= 0:
            raise ValueError("Population must be positive")
        return v

    @field_validator("gdp_per_capita")
    def validate_gdp(cls, v):
        if v < 0:
            raise ValueError("GDP per capita cannot be negative")
        return v


# =============================================================================
# SQLMODEL DATABASE MODELS WITH PROPER RELATIONSHIPS
# =============================================================================


class Country(SQLModel, table=True):
    __name__ = "country"  # Singular table name
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    country: str = Field(index=True)
    sub_region: str
    code: str = Field(unique=True, index=True)
    area_km2: Optional[float] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EconomicData(SQLModel, table=True):
    __name__ = "economic_data"  # Singular table name
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    country_id: int = Field(
        foreign_key="country.id", index=True
    )  # Foreign key relationship
    year: int = Field(index=True)
    population: int
    gdp_per_capita: float
    gdp_per_capita_ppp: Optional[float] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EmissionData(SQLModel, table=True):
    __name__ = "emission_data"  # Singular table name
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    country_id: int = Field(
        foreign_key="country.id", index=True
    )  # Foreign key relationship
    year: int = Field(index=True)
    transportation_mt: Optional[float] = 0.0
    total_co2_including_lucf: Optional[float] = None
    total_co2_excluding_lucf: float
    other_fuel_combustion: Optional[float] = 0.0
    manufacturing_construction: Optional[float] = 0.0
    land_use_change_forestry: Optional[float] = 0.0
    industrial_processes: Optional[float] = 0.0
    energy_mt: Optional[float] = 0.0
    electricity_heat: Optional[float] = 0.0
    bunker_fuels: Optional[float] = 0.0
    building_mt: Optional[float] = 0.0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# DATABASE CONNECTION AND OPERATIONS
# =============================================================================

DATABASE_URL = os.environ["DATABASE_URL"]
# Global engine variable
engine = None


def get_engine():
    """Get or create database engine"""
    global engine
    if engine is None:
        engine = create_engine(DATABASE_URL, echo=False)
    return engine


def init_database():
    """Initialize database tables - called once at app startup"""
    try:
        engine = get_engine()
        print("Creating database tables...")
        SQLModel.metadata.create_all(engine)
        print("Database tables created successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        return False


# =============================================================================
# LOADING DATA FROM CSVs WITH PROPER RELATIONSHIPS AND ERROR HANDLING
# =============================================================================


def inspect_csv_columns(csv_path: str):
    """Inspect CSV columns for debugging"""
    try:
        df = pd.read_csv(
            csv_path, nrows=1
        )  # Read only first row to check columns
        print(f"Columns in {csv_path}: {list(df.columns)}")
        return list(df.columns)
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        return []


def load_countries(csv_path: str) -> dict[str, int]:
    """Load countries and return mapping of country name to ID"""
    # First inspect the CSV structure
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    country_map: dict[str, int] = {}

    # Print first few rows for debugging
    print(f"First 3 rows of {csv_path}:")
    print(df.head(3))

    # Try to identify the correct column names (case-insensitive)
    country_col = None
    sub_region_col = None
    code_col = None
    area_col = None

    # Look for country column variations
    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(
            x in col_lower for x in ["code", "id"]
        ):
            country_col = col
        elif "sub" in col_lower and "region" in col_lower:
            sub_region_col = col
        elif "code" in col_lower:
            code_col = col
        elif "area" in col_lower or "km" in col_lower:
            area_col = col

    print(
        f"Identified columns - Country: {country_col}, Sub-region: {sub_region_col}, Code: {code_col}, Area: {area_col}"
    )

    if not all([country_col, sub_region_col, code_col]):
        print("Error: Could not identify required columns in countries.csv")
        return {}

    with Session(get_engine()) as session:
        for _, row in df.iterrows():
            try:
                country = Country(
                    country=str(row[country_col]),
                    sub_region=str(row[sub_region_col]),
                    code=str(row[code_col]),
                    area_km2=row.get(area_col) if area_col else None,
                )
                session.add(country)
            except Exception as e:
                print(f"Error processing country row: {e}")
                continue

        session.commit()

        # Build map from country name to id
        stmt = select(Country)
        for c in session.exec(stmt):
            country_map[c.country] = c.id  # type: ignore

    print(f"Loaded {len(country_map)} countries")
    return country_map


def load_economic_data(csv_path: str, country_map: dict[str, int]):
    """Load economic data with proper country_id relationships"""
    # First inspect the CSV structure
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Print first few rows for debugging
    print(f"First 3 rows of {csv_path}:")
    print(df.head(3))

    # Try to identify the correct column names
    country_col = None
    year_col = None
    population_col = None
    gdp_col = None
    gdp_ppp_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(
            x in col_lower for x in ["code", "id"]
        ):
            country_col = col
        elif "year" in col_lower:
            year_col = col
        elif "population" in col_lower:
            population_col = col
        elif "gdp" in col_lower and "ppp" not in col_lower:
            gdp_col = col
        elif "gdp" in col_lower and "ppp" in col_lower:
            gdp_ppp_col = col

    print(
        f"Identified columns - Country: {country_col}, Year: {year_col}, Population: {population_col}, GDP: {gdp_col}, GDP PPP: {gdp_ppp_col}"
    )

    if not all([country_col, year_col, population_col, gdp_col]):
        print("Error: Could not identify required columns in economic_data.csv")
        return

    with Session(get_engine()) as session:
        success_count = 0
        error_count = 0

        for _, row in df.iterrows():
            try:
                country_name = str(row[country_col])
                country_id = country_map.get(country_name)

                if country_id is None:
                    print(
                        f"Warning: Country '{country_name}' not found in country_map"
                    )
                    error_count += 1
                    continue

                econ = EconomicData(
                    country_id=country_id,
                    year=int(row[year_col]),
                    population=int(row[population_col]),
                    gdp_per_capita=float(row[gdp_col]),
                    gdp_per_capita_ppp=row.get(gdp_ppp_col)
                    if gdp_ppp_col
                    else None,
                )
                session.add(econ)
                success_count += 1

            except Exception as e:
                print(f"Error processing economic data row: {e}")
                error_count += 1
                continue

        session.commit()
        print(f"Economic data: {success_count} success, {error_count} errors")


def insert_country_data(data: CountryData):
    """Insert validated data into database tables using SQLModel with proper relationships"""
    try:
        engine = get_engine()
        with Session(engine) as session:
            # Check if country exists first
            country_stmt = select(Country).where(Country.code == data.code)
            existing_country = session.exec(country_stmt).first()

            if not existing_country:
                # Insert new country
                country = Country(
                    country=data.country,
                    sub_region=data.sub_region,
                    code=data.code,
                    area_km2=data.area_km2,
                )
                session.add(country)
                session.commit()  # Commit to get the ID
                session.refresh(country)  # Refresh to get the generated ID
                country_id = country.id
            else:
                country_id = existing_country.id

            # Ensure country_id is not None before proceeding
            if country_id is None:
                raise ValueError("Failed to get valid country_id")

            # Check if economic data exists for this country/year
            economic_stmt = select(EconomicData).where(
                EconomicData.country_id == country_id,
                EconomicData.year == data.year,
            )
            existing_economic = session.exec(economic_stmt).first()

            if not existing_economic:
                # Insert economic data
                economic = EconomicData(
                    country_id=country_id,  # Now guaranteed to be an int, not None
                    year=data.year,
                    population=data.population,
                    gdp_per_capita=data.gdp_per_capita,
                    gdp_per_capita_ppp=data.gdp_per_capita_ppp,
                )
                session.add(economic)

            # Check if emission data exists for this country/year
            emission_stmt = select(EmissionData).where(
                EmissionData.country_id == country_id,
                EmissionData.year == data.year,
            )
            existing_emission = session.exec(emission_stmt).first()

            if not existing_emission:
                # Insert emission data
                emission = EmissionData(
                    country_id=country_id,  # Now guaranteed to be an int, not None
                    year=data.year,
                    transportation_mt=data.transportation_mt,
                    total_co2_including_lucf=data.total_co2_including_lucf,
                    total_co2_excluding_lucf=data.total_co2_excluding_lucf,
                    other_fuel_combustion=data.other_fuel_combustion,
                    manufacturing_construction=data.manufacturing_construction,
                    land_use_change_forestry=data.land_use_change_forestry,
                    industrial_processes=data.industrial_processes,
                    energy_mt=data.energy_mt,
                    electricity_heat=data.electricity_heat,
                    bunker_fuels=data.bunker_fuels,
                    building_mt=data.building_mt,
                )
                session.add(emission)

            session.commit()
            return True

    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False


def load_emission_data(csv_path: str, country_map: dict[str, int]):
    """Load emission data with proper country_id relationships"""
    # First inspect the CSV structure
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Print first few rows for debugging
    print(f"First 3 rows of {csv_path}:")
    print(df.head(3))

    # Try to identify the correct column names
    country_col = None
    year_col = None
    co2_excl_col = None
    co2_incl_col = None
    energy_col = None
    transport_col = None
    electricity_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(
            x in col_lower for x in ["code", "id"]
        ):
            country_col = col
        elif "year" in col_lower:
            year_col = col
        elif "co2" in col_lower and "excluding" in col_lower:
            co2_excl_col = col
        elif "co2" in col_lower and "including" in col_lower:
            co2_incl_col = col
        elif "energy" in col_lower and "mt" in col_lower:
            energy_col = col
        elif "transport" in col_lower:
            transport_col = col
        elif "electricity" in col_lower or "heat" in col_lower:
            electricity_col = col

    print(
        f"Identified columns - Country: {country_col}, Year: {year_col}, CO2 Excl: {co2_excl_col}"
    )

    if not all([country_col, year_col, co2_excl_col]):
        print("Error: Could not identify required columns in emissions.csv")
        return

    with Session(get_engine()) as session:
        success_count = 0
        error_count = 0

        for _, row in df.iterrows():
            try:
                country_name = str(row[country_col])
                country_id = country_map.get(country_name)

                if country_id is None:
                    print(
                        f"Warning: Country '{country_name}' not found in country_map"
                    )
                    error_count += 1
                    continue

                emis = EmissionData(
                    country_id=country_id,
                    year=int(row[year_col]),
                    total_co2_excluding_lucf=float(row[co2_excl_col]),
                    total_co2_including_lucf=row.get(co2_incl_col)
                    if co2_incl_col
                    else None,
                    energy_mt=row.get(energy_col) if energy_col else None,
                    transportation_mt=row.get(transport_col)
                    if transport_col
                    else None,
                    electricity_heat=row.get(electricity_col)
                    if electricity_col
                    else None,
                    # Set other fields to default values for now
                    other_fuel_combustion=0.0,
                    manufacturing_construction=0.0,
                    land_use_change_forestry=0.0,
                    industrial_processes=0.0,
                    bunker_fuels=0.0,
                    building_mt=0.0,
                )
                session.add(emis)
                success_count += 1

            except Exception as e:
                print(f"Error processing emission data row: {e}")
                error_count += 1
                continue

        session.commit()
        print(f"Emission data: {success_count} success, {error_count} errors")


def get_all_data():
    """Fetch all data for visualization using SQLModel joins with proper relationships"""
    try:
        engine = get_engine()
        with Session(engine) as session:
            # Use SQLModel select with proper joins using country_id
            stmt = (
                select(Country, EconomicData, EmissionData)
                .select_from(Country)
                .join(EconomicData, Country.id == EconomicData.country_id)
                .join(
                    EmissionData,
                    and_(
                        Country.id == EmissionData.country_id,
                        EconomicData.year == EmissionData.year,
                    ),
                )
                .order_by(Country.country, EconomicData.year)
            )

            results = session.exec(stmt).all()

            if results:
                # Convert results to DataFrame
                data_list = []
                for row in results:
                    country, economic, emission = row
                    data_list.append(
                        {
                            "country": country.country,
                            "sub_region": country.sub_region,
                            "code": country.code,
                            "area_km2": country.area_km2,
                            "year": economic.year,
                            "population": economic.population,
                            "gdp_per_capita": economic.gdp_per_capita,
                            "gdp_per_capita_ppp": economic.gdp_per_capita_ppp,
                            "total_co2_excluding_lucf": emission.total_co2_excluding_lucf,
                            "total_co2_including_lucf": emission.total_co2_including_lucf,
                            "energy_mt": emission.energy_mt,
                            "transportation_mt": emission.transportation_mt,
                            "electricity_heat": emission.electricity_heat,
                            "manufacturing_construction": emission.manufacturing_construction,
                            "industrial_processes": emission.industrial_processes,
                        }
                    )

                return pd.DataFrame(data_list)

            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


# =============================================================================
# DATA PROCESSING PIPELINE
# =============================================================================


class DataPipeline:
    """Complete data processing pipeline"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Load trained ML model"""
        try:
            self.model = joblib.load("co2_emission_model.pkl")
        except FileNotFoundError:
            st.warning("ML model not found. Please train the model first.")

    async def process_csv_data(self, uploaded_file):
        """Process uploaded CSV data"""
        try:
            # Read CSV with polars for better performance
            df = pl.read_csv(
                uploaded_file, null_values=["", " ", "NA", "N/A", "nan", "NaN"]
            )

            # Data cleaning pipeline
            df = self.clean_data(df)

            # Validate and insert data
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
        """Clean and transform data"""
        # Remove columns with >50% nulls
        threshold = 0.5
        null_counts = df.null_count()
        columns_to_drop = [
            col
            for col, null_count in zip(df.columns, null_counts.row(0))
            if null_count / df.height > threshold
        ]

        if columns_to_drop:
            df = df.drop(columns_to_drop)

        # Fill nulls with median for numeric columns
        numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        df = df.with_columns(
            [
                pl.col(col).fill_null(pl.col(col).median())
                for col in numeric_cols
            ]
        )

        # Remove outliers using IQR
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
        """Predict CO2 emissions using trained model"""
        if self.model is None:
            return None

        try:
            features = np.array([[population, gdp_per_capita, energy_mt]])
            prediction = self.model.predict(features)[0]
            return max(0, prediction)  # Ensure non-negative prediction
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================


def create_sidebar():
    """Create sidebar with tab navigation"""
    st.sidebar.markdown("# 🌍 CO2 Emissions Pipeline")
    st.sidebar.markdown("---")

    # Create button tabs in sidebar
    tabs = [
        "Dashboard",
        "Data Input",
        "Bulk Upload",
        "Predictions",
        "Analytics",
    ]

    # Initialize session state for active tab if not exists
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dashboard"

    st.sidebar.markdown("### Navigation")

    # Create buttons for each tab
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
    """Create data input form"""
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
    """Create main dashboard with visualizations and explanations"""
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

    # Key metrics with explanations
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

    # Time series visualization
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

    # Regional comparison
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

    # Top emitters for latest year
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

    # GDP vs Emissions scatter plot
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

    # Correlation analysis
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
    """Create prediction interface"""
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

                    # Prediction context
                    st.markdown("**📋 Input Summary:**")
                    st.write(f"• Population: {pred_population:,}")
                    st.write(f"• GDP Per Capita: ${pred_gdp:,.2f}")
                    st.write(f"• Energy Consumption: {pred_energy:.2f} Mt")

                    # Additional insights
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


# =============================================================================
# APPLICATION INITIALIZATION WITH BETTER ERROR HANDLING
# =============================================================================


@st.cache_resource
def initialize_app():
    """Initialize the application - called once at startup"""
    with st.spinner("🔧 Initializing database and pipeline..."):
        success = init_database()
        if not success:
            st.error("❌ Failed to initialize database!")
            st.stop()

        # Load data with proper relationships
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory {data_dir} does not exist")
            st.warning(
                "⚠️ Data directory not found. Please ensure CSV files are in the 'data' folder."
            )
            pipeline = DataPipeline()
            return pipeline

        # Check for required CSV files
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
            # First load countries to get the mapping
            print("Loading countries...")
            country_map = load_countries(
                os.path.join(data_dir, "countries.csv")
            )

            if not country_map:
                print("Failed to load countries")
                st.error("❌ Failed to load countries data")
                pipeline = DataPipeline()
                return pipeline

            # Then load economic and emission data using the country mapping
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

    # Initialize pipeline
    pipeline = DataPipeline()
    return pipeline


# =============================================================================
# MAIN APPLICATION
# =============================================================================


async def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="CO2 Emissions Pipeline",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
   <style>
   .stButton > button {
      width: 100%;
      margin-bottom: 5px;
   }
   .metric-container {
      background-color: #f0f2f6;
      padding: 10px;
      border-radius: 5px;
      margin: 5px 0;
   }
   </style>
   """,
        unsafe_allow_html=True,
    )

    # Initialize application
    pipeline = initialize_app()

    # Create sidebar navigation
    page = create_sidebar()

    # Load data
    df = get_all_data()

    # Route to appropriate page
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
            # Enhanced analytics section
            tab1, tab2, tab3 = st.tabs(
                ["📊 Statistical Summary", "🔍 Data Explorer", "📋 Raw Data"]
            )

            with tab1:
                st.subheader("Statistical Overview")
                st.dataframe(df.describe(), use_container_width=True)

                # Additional statistics
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

                # Filters
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

                # Filter data
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

                # Download option
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
