from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, field_validator
from sqlmodel import Field, SQLModel

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
