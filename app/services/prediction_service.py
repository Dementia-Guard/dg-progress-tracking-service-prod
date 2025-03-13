import numpy as np
import pandas as pd
from app.core.model_manager import model_manager
from app.schemas.prediction import PredictionRequest, PredictedVisit, ProgressionRate, PredictionResponse


class PredictionService:
    @staticmethod
    def predict_progression(request: PredictionRequest) -> PredictionResponse:
        """
        Predict dementia progression based on previous visits

        Args:
            request: The prediction request with visit data

        Returns:
            PredictionResponse: Predicted future visits and progression rate
        """
        # Get models
        mmse_model = model_manager.get_mmse_model()
        cdr_model = model_manager.get_cdr_model()

        # Extract sequence data from visits
        visits_data = []
        for visit in request.visits:
            visits_data.append([visit.mmse, visit.cdr, visit.age])

        input_sequence = np.array(visits_data)

        # Make predictions
        predictions = PredictionService._predict_progression(
            mmse_model,
            cdr_model,
            input_sequence,
            num_future_visits=request.num_future_visits,
            interval_months=request.interval_months
        )

        # Convert to response model
        return PredictionService._create_response(predictions, request)

    @staticmethod
    def _predict_progression(
            model_mmse,
            model_cdr,
            input_sequence,
            num_future_visits=3,
            interval_months=6
    ) -> pd.DataFrame:
        """
        Predict future MMSE and CDR values based on input sequence

        Args:
            model_mmse: Trained LSTM model for MMSE prediction
            model_cdr: Trained LSTM model for CDR prediction
            input_sequence: numpy array of shape (seq_length, 3) with MMSE, CDR, and Age values
            num_future_visits: Number of future visits to predict
            interval_months: Months between visits (for age calculation)

        Returns:
            predictions: DataFrame with predicted MMSE and CDR values
        """
        # Make a copy of the input sequence
        current_sequence = input_sequence.copy()

        # Initialize results
        results = []

        # Store the latest values
        latest_mmse = current_sequence[-1, 0]
        latest_cdr = current_sequence[-1, 1]
        latest_age = current_sequence[-1, 2]

        # Add baseline entry
        results.append({
            'Visit': 'Latest',
            'MMSE': latest_mmse,
            'CDR': latest_cdr,
            'Age': latest_age,
            'MMSE_Change': 0,
            'CDR_Change': 0
        })

        # Predict future visits
        for i in range(num_future_visits):
            # Predict change in MMSE and CDR
            mmse_change = model_mmse.predict(np.expand_dims(current_sequence, axis=0), verbose=0)[0][0]
            cdr_change = model_cdr.predict(np.expand_dims(current_sequence, axis=0), verbose=0)[0][0]

            # Calculate new values
            new_mmse = latest_mmse + mmse_change
            new_cdr = latest_cdr + cdr_change
            new_age = latest_age + (interval_months / 12.0)  # Convert months to years

            # Ensure values are within clinical bounds
            new_mmse = max(0, min(30, new_mmse))  # MMSE ranges from 0 to 30
            new_cdr = max(0, min(3, new_cdr))  # CDR usually ranges from 0 to 3

            # Store results
            results.append({
                'Visit': f'Predicted {i + 1}',
                'MMSE': new_mmse,
                'CDR': new_cdr,
                'Age': new_age,
                'MMSE_Change': mmse_change,
                'CDR_Change': cdr_change
            })

            # Update latest values
            latest_mmse = new_mmse
            latest_cdr = new_cdr
            latest_age = new_age

            # Update sequence for next prediction
            new_sequence = np.array([[new_mmse, new_cdr, new_age]])
            current_sequence = np.vstack([current_sequence[1:], new_sequence])

        return pd.DataFrame(results)

    @staticmethod
    def _create_response(predictions: pd.DataFrame, request: PredictionRequest) -> PredictionResponse:
        """Convert predictions DataFrame to API response model"""
        # Create baseline visit
        baseline = PredictedVisit(
            visit_number=0,
            mmse=float(predictions.iloc[0]['MMSE']),
            cdr=float(predictions.iloc[0]['CDR']),
            age=float(predictions.iloc[0]['Age']),
            mmse_change=0.0,
            cdr_change=0.0
        )

        # Create predicted visits
        predicted_visits = []
        for i in range(1, len(predictions)):
            visit = predictions.iloc[i]
            predicted_visits.append(PredictedVisit(
                visit_number=i,
                mmse=float(visit['MMSE']),
                cdr=float(visit['CDR']),
                age=float(visit['Age']),
                mmse_change=float(visit['MMSE_Change']),
                cdr_change=float(visit['CDR_Change'])
            ))

        # Calculate progression rates
        progression_rate = PredictionService._calculate_progression_rate(predictions)

        # Create response
        return PredictionResponse(
            baseline=baseline,
            predictions=predicted_visits,
            progression_rate=progression_rate
        )

    @staticmethod
    def _calculate_progression_rate(predictions: pd.DataFrame) -> ProgressionRate:
        """Calculate annual progression rates and clinical interpretation"""
        if len(predictions) > 1:
            first_pred = predictions.iloc[0]
            last_pred = predictions.iloc[-1]
            years_diff = last_pred['Age'] - first_pred['Age']

            if years_diff > 0:
                mmse_annual_change = (last_pred['MMSE'] - first_pred['MMSE']) / years_diff
                cdr_annual_change = (last_pred['CDR'] - first_pred['CDR']) / years_diff
            else:
                mmse_annual_change = 0
                cdr_annual_change = 0

            # Interpret results
            mmse_trajectory = "stable" if abs(
                mmse_annual_change) < 1 else "declining" if mmse_annual_change < 0 else "improving"
            cdr_trajectory = "stable" if abs(
                cdr_annual_change) < 0.2 else "worsening" if cdr_annual_change > 0 else "improving"
            rapid_progression = mmse_annual_change < -2 or cdr_annual_change > 0.5

            return ProgressionRate(
                mmse_annual_change=float(mmse_annual_change),
                cdr_annual_change=float(cdr_annual_change),
                mmse_trajectory=mmse_trajectory,
                cdr_trajectory=cdr_trajectory,
                rapid_progression=rapid_progression
            )
        else:
            # Default values if not enough data
            return ProgressionRate(
                mmse_annual_change=0.0,
                cdr_annual_change=0.0,
                mmse_trajectory="unknown",
                cdr_trajectory="unknown",
                rapid_progression=False
            )
