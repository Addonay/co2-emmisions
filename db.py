import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import and_
from sqlmodel import Session, SQLModel, create_engine, select
import numpy as np

from models import Country, CountryData, EconomicData, EmissionData

load_dotenv()

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
# UTILITY FUNCTIONS
# =============================================================================

def safe_int_convert(value):
    """Safely convert value to int, handling NaN and empty strings"""
    if pd.isna(value) or value == '' or value is None:
        return None
    try:
        return int(float(value))  # Convert to float first to handle string floats
    except (ValueError, TypeError):
        return None


def safe_float_convert(value):
    """Safely convert value to float, handling NaN and empty strings"""
    if pd.isna(value) or value == '' or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_str_convert(value):
    """Safely convert value to string, handling NaN"""
    if pd.isna(value):
        return None
    return str(value).strip()


# =============================================================================
# LOADING DATA FROM CSVs WITH PROPER RELATIONSHIPS AND ERROR HANDLING
# =============================================================================

def inspect_csv_columns(csv_path: str):
    """Inspect CSV columns for debugging"""
    try:
        df = pd.read_csv(csv_path, nrows=1)
        print(f"Columns in {csv_path}: {list(df.columns)}")
        return list(df.columns)
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        return []


def load_countries(csv_path: str) -> dict[str, int]:
    """Load countries with optimized duplicate checking"""
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    country_map: dict[str, int] = {}
    print(f"Processing {len(df)} rows from {csv_path}")
    
    # Column identification
    country_col = None
    sub_region_col = None
    code_col = None
    area_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(x in col_lower for x in ["code", "id"]):
            country_col = col
        elif "sub" in col_lower and "region" in col_lower:
            sub_region_col = col
        elif "code" in col_lower:
            code_col = col
        elif "area" in col_lower or "km" in col_lower:
            area_col = col
            
    print(f"Identified columns - Country: {country_col}, Sub-region: {sub_region_col}, Code: {code_col}, Area: {area_col}")

    if not all([country_col, sub_region_col, code_col]):
        print("Error: Could not identify required columns (Country, Sub-region, Code)")
        return {}

    with Session(get_engine()) as session:
        # Get all existing countries at once
        existing_countries_stmt = select(Country)
        existing_countries = session.exec(existing_countries_stmt).all()
        existing_codes = {c.code for c in existing_countries}
        
        # Build country map from existing countries
        for c in existing_countries:
            country_map[c.country] = c.id
        
        print(f"Found {len(existing_codes)} existing countries in database")

        # Process new countries in batch
        new_countries = []
        duplicates_skipped = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                country_name = safe_str_convert(row[country_col])
                country_code = safe_str_convert(row[code_col])
                sub_region = safe_str_convert(row[sub_region_col])
                area_km2 = safe_float_convert(row.get(area_col)) if area_col else None

                if not country_name or not country_code:
                    errors += 1
                    continue

                if country_code in existing_codes:
                    duplicates_skipped += 1
                    continue

                country = Country(
                    country=country_name,
                    sub_region=sub_region,
                    code=country_code,
                    area_km2=area_km2,
                )
                new_countries.append(country)
                existing_codes.add(country_code)  # Prevent duplicates within the same batch

            except Exception as e:
                errors += 1
                continue

        # Batch insert new countries
        if new_countries:
            session.add_all(new_countries)
            session.commit()
            
            # Refresh to get IDs and update country_map
            for country in new_countries:
                session.refresh(country)
                country_map[country.country] = country.id

        print(f"Countries: {len(new_countries)} added, {duplicates_skipped} duplicates skipped, {errors} errors")
    
    return country_map


def load_economic_data(csv_path: str, country_map: dict[str, int]):
    """Load economic data with optimized duplicate checking"""
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} rows from {csv_path}")
    
    # Column identification
    country_col = None
    year_col = None
    population_col = None
    gdp_col = None
    gdp_ppp_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(x in col_lower for x in ["code", "id"]):
            country_col = col
        elif "year" in col_lower:
            year_col = col
        elif "population" in col_lower:
            population_col = col
        elif "gdp" in col_lower and "ppp" not in col_lower:
            gdp_col = col
        elif "gdp" in col_lower and "ppp" in col_lower:
            gdp_ppp_col = col

    if not gdp_col and gdp_ppp_col:
        gdp_col = gdp_ppp_col

    print(f"Identified columns - Country: {country_col}, Year: {year_col}, Population: {population_col}, GDP: {gdp_col}")

    if not all([country_col, year_col, population_col, gdp_col]):
        print("Error: Could not identify required columns")
        return

    with Session(get_engine()) as session:
        # Get existing economic data combinations
        existing_stmt = select(EconomicData.country_id, EconomicData.year)
        existing_combinations = {(row[0], row[1]) for row in session.exec(existing_stmt).all()}
        
        print(f"Found {len(existing_combinations)} existing economic data records")

        new_records = []
        duplicates_skipped = 0
        country_not_found = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                country_name = safe_str_convert(row[country_col])
                year = safe_int_convert(row[year_col])
                population = safe_int_convert(row[population_col])
                gdp_per_capita = safe_float_convert(row[gdp_col])
                gdp_per_capita_ppp = safe_float_convert(row.get(gdp_ppp_col)) if gdp_ppp_col else None

                if not country_name or year is None:
                    errors += 1
                    continue

                country_id = country_map.get(country_name)
                if country_id is None:
                    country_not_found += 1
                    continue

                if (country_id, year) in existing_combinations:
                    duplicates_skipped += 1
                    continue

                if population is None or gdp_per_capita is None:
                    errors += 1
                    continue

                econ = EconomicData(
                    country_id=country_id,
                    year=year,
                    population=population,
                    gdp_per_capita=gdp_per_capita,
                    gdp_per_capita_ppp=gdp_per_capita_ppp,
                )
                new_records.append(econ)
                existing_combinations.add((country_id, year))

            except Exception as e:
                errors += 1
                continue

        # Batch insert
        if new_records:
            session.add_all(new_records)
            session.commit()

        print(f"Economic data: {len(new_records)} added, {duplicates_skipped} duplicates, {country_not_found} country not found, {errors} errors")


def load_emission_data(csv_path: str, country_map: dict[str, int]):
    """Load emission data with optimized duplicate checking"""
    columns = inspect_csv_columns(csv_path)
    if not columns:
        print(f"Could not read columns from {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} rows from {csv_path}")
    
    # Column identification
    country_col = None
    year_col = None
    co2_excl_col = None
    co2_incl_col = None
    energy_col = None
    transport_col = None
    electricity_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if "country" in col_lower and not any(x in col_lower for x in ["code", "id"]):
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

    print(f"Identified columns - Country: {country_col}, Year: {year_col}, CO2 Excl: {co2_excl_col}")

    if not all([country_col, year_col, co2_excl_col]):
        print("Error: Could not identify required columns")
        return

    with Session(get_engine()) as session:
        # Get existing emission data combinations
        existing_stmt = select(EmissionData.country_id, EmissionData.year)
        existing_combinations = {(row[0], row[1]) for row in session.exec(existing_stmt).all()}
        
        print(f"Found {len(existing_combinations)} existing emission data records")

        new_records = []
        duplicates_skipped = 0
        country_not_found = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                country_name = safe_str_convert(row[country_col])
                year = safe_int_convert(row[year_col])
                total_co2_excluding_lucf = safe_float_convert(row[co2_excl_col])

                if not country_name or year is None or total_co2_excluding_lucf is None:
                    errors += 1
                    continue

                country_id = country_map.get(country_name)
                if country_id is None:
                    country_not_found += 1
                    continue

                if (country_id, year) in existing_combinations:
                    duplicates_skipped += 1
                    continue

                emis = EmissionData(
                    country_id=country_id,
                    year=year,
                    total_co2_excluding_lucf=total_co2_excluding_lucf,
                    total_co2_including_lucf=safe_float_convert(row.get(co2_incl_col)) if co2_incl_col else None,
                    energy_mt=safe_float_convert(row.get(energy_col)) if energy_col else None,
                    transportation_mt=safe_float_convert(row.get(transport_col)) if transport_col else None,
                    electricity_heat=safe_float_convert(row.get(electricity_col)) if electricity_col else None,
                    other_fuel_combustion=0.0,
                    manufacturing_construction=0.0,
                    land_use_change_forestry=0.0,
                    industrial_processes=0.0,
                    bunker_fuels=0.0,
                    building_mt=0.0,
                )
                new_records.append(emis)
                existing_combinations.add((country_id, year))

            except Exception as e:
                errors += 1
                continue

        # Batch insert
        if new_records:
            session.add_all(new_records)
            session.commit()

        print(f"Emission data: {len(new_records)} added, {duplicates_skipped} duplicates, {country_not_found} country not found, {errors} errors")


def insert_country_data(data: CountryData):
    """Insert single country data record"""
    try:
        engine = get_engine()
        with Session(engine) as session:
            # Check if country exists
            country_stmt = select(Country).where(Country.code == data.code)
            existing_country = session.exec(country_stmt).first()
            
            country_id = None
            if not existing_country:
                country = Country(
                    country=data.country,
                    sub_region=data.sub_region,
                    code=data.code,
                    area_km2=data.area_km2,
                )
                session.add(country)
                session.commit()
                session.refresh(country)
                country_id = country.id
            else:
                country_id = existing_country.id

            if country_id is None:
                raise ValueError("Failed to get or create a valid country_id")

            # Check and insert economic data
            economic_stmt = select(EconomicData).where(
                EconomicData.country_id == country_id,
                EconomicData.year == data.year,
            )
            existing_economic = session.exec(economic_stmt).first()
            if not existing_economic and data.population is not None and data.gdp_per_capita is not None:
                economic = EconomicData(
                    country_id=country_id,
                    year=data.year,
                    population=data.population,
                    gdp_per_capita=data.gdp_per_capita,
                    gdp_per_capita_ppp=data.gdp_per_capita_ppp,
                )
                session.add(economic)

            # Check and insert emission data
            emission_stmt = select(EmissionData).where(
                EmissionData.country_id == country_id,
                EmissionData.year == data.year,
            )
            existing_emission = session.exec(emission_stmt).first()
            if not existing_emission and data.total_co2_excluding_lucf is not None:
                emission = EmissionData(
                    country_id=country_id,
                    year=data.year,
                    transportation_mt=data.transportation_mt or 0.0,
                    total_co2_including_lucf=data.total_co2_including_lucf,
                    total_co2_excluding_lucf=data.total_co2_excluding_lucf,
                    other_fuel_combustion=data.other_fuel_combustion or 0.0,
                    manufacturing_construction=data.manufacturing_construction or 0.0,
                    land_use_change_forestry=data.land_use_change_forestry or 0.0,
                    industrial_processes=data.industrial_processes or 0.0,
                    energy_mt=data.energy_mt or 0.0,
                    electricity_heat=data.electricity_heat or 0.0,
                    bunker_fuels=data.bunker_fuels or 0.0,
                    building_mt=data.building_mt or 0.0,
                )
                session.add(emission)
                
            session.commit()
            return True
    except Exception as e:
        session.rollback()
        st.error(f"Database error during insert: {str(e)}")
        return False


def get_all_data():
    """Retrieve all data with proper error handling"""
    try:
        engine = get_engine()
        with Session(engine) as session:
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
                data_list = []
                for row in results:
                    country, economic, emission = row
                    data_list.append({
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
                    })
                return pd.DataFrame(data_list)
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()