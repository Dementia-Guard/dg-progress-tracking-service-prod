from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.core.model_manager import model_manager
import logging

router = APIRouter(tags=["predictions"])
logger = logging.getLogger(__name__)


def validate_models():
    """Dependency to check if models are loaded"""
    if not model_manager.is_initialized():
        success = model_manager.initialize()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please check server logs."
            )
    return True


@router.post("/predict", response_model=PredictionResponse)
async def predict_progression(
        request: PredictionRequest,
        _: bool = Depends(validate_models)
):
    """
    Predict future dementia progression based on previous visit data

    - **visits**: List of previous visits with MMSE, CDR, and age values
    - **num_future_visits**: Number of future visits to predict (default: 3)
    - **interval_months**: Months between visits (default: 6)

    Returns predicted visits and progression rate analysis
    """
    try:
        logger.info(f"Received prediction request for {len(request.visits)} visits")

        # Validate sequence length
        if len(request.visits) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 previous visits are required for prediction"
            )

        # Get predictions
        result = PredictionService.predict_progression(request)
        logger.info("Prediction successful")

        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )
