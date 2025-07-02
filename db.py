import os

import pandas as pd
import streamlit as st
from sqlalchemy import and_
from sqlmodel import Session, SQLModel, create_engine, select

from models import Country, CountryData, EconomicData, EmissionData

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
