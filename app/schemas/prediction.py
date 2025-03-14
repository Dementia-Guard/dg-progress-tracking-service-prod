from pydantic import BaseModel, Field, validator
from typing import List, Optional


class Visit(BaseModel):
    """Data for a single patient visit"""
    mmse: float = Field(..., ge=0, le=30, description="MMSE score (0-30)")
    cdr: float = Field(..., ge=0, le=3, description="CDR score (0, 0.5, 1, 2, 3)")
    age: float = Field(..., ge=0, le=120, description="Patient age in years")

    @validator('cdr')
    def check_cdr_value(cls, v):
        valid_values = [0, 0.5, 1, 2, 3]
        if v not in valid_values:
            raise ValueError(f'CDR must be one of {valid_values}')
        return v


class PredictionRequest(BaseModel):
    """Request model for progression prediction"""
    visits: List[Visit] = Field(..., min_items=2, description="Previous patient visits (at least 2)")
    num_future_visits: int = Field(3, ge=1, le=10, description="Number of future visits to predict")
    interval_months: int = Field(6, ge=1, le=24, description="Months between predicted visits")


class PredictedVisit(BaseModel):
    """Data for a predicted visit"""
    visit_number: int
    mmse: float
    cdr: float
    age: float
    mmse_change: Optional[float] = None
    cdr_change: Optional[float] = None


class ProgressionRate(BaseModel):
    """Annual progression rates"""
    mmse_annual_change: float
    cdr_annual_change: float
    mmse_trajectory: str
    cdr_trajectory: str
    rapid_progression: bool


class PredictionResponse(BaseModel):
    """Response model with predictions"""
    baseline: PredictedVisit
    predictions: List[PredictedVisit]
    progression_rate: ProgressionRate
