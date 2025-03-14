import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class to load and manage Models
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.mmse_model = None
            cls._instance.cdr_model = None
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        """Load models from disk"""
        try:
            logger.info(f"Loading MMSE model from {settings.MMSE_MODEL_PATH}")
            self.mmse_model = tf.keras.models.load_model(settings.MMSE_MODEL_PATH)

            logger.info(f"Loading CDR model from {settings.CDR_MODEL_PATH}")
            self.cdr_model = tf.keras.models.load_model(settings.CDR_MODEL_PATH)

            self.initialized = True
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def is_initialized(self):
        """Check if models are loaded"""
        return self.initialized

    def get_mmse_model(self):
        """Get the MMSE prediction model"""
        if not self.initialized:
            self.initialize()
        return self.mmse_model

    def get_cdr_model(self):
        """Get the CDR prediction model"""
        if not self.initialized:
            self.initialize()
        return self.cdr_model

model_manager = ModelManager()