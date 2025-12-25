#!/usr/bin/env python3
"""
Configuration settings for Pixtral Mistral Integration Package
"""

# Mistral API Configuration
DEFAULT_PIXTRAL_MODEL = "pixtral-12b-2409"
DEFAULT_CHAT_MODEL = "mistral-small-latest"

# Analysis Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMP_DIR = "temp_analysis"

# Object Analysis Prompts
OBJECT_ANALYSIS_PROMPTS = {
    "default": "Describe this {class_name} in one simple sentence focusing on its appearance and condition in 10 words. Mention if the item is from any particular brand sign on it",
    "detailed": "Provide a detailed description of this {class_name} including its appearance, condition, and any visible text or branding",
    "simple": "Describe this {class_name} in one short sentence",
    "brand_focused": "Identify any visible branding, text, or logos on this {class_name}",
    "condition": "Describe the condition and appearance of this {class_name}"
}

# Scene Analysis Prompts
SCENE_ANALYSIS_PROMPTS = {
    "default": "Describe the scene in 2-3 concise sentences",
    "detailed": "Provide a comprehensive description of the scene including all visible objects and their relationships",
    "simple": "What do you see in this image?",
    "objects_only": "List all the objects you can see in this scene",
    "document_detection": "Look for any documents, papers, books, magazines, or text-based materials in this scene. Describe what you see, focusing on any written or printed materials.",
    "concise": "Describe this image in ONE SHORT SENTENCE (maximum 15 words). Be specific and factual."
}

# OCR Configuration
OCR_PROMPTS = {
    "default": "Extract all visible text from this image",
    "detailed": "Extract all visible text from this image and describe the context",
    "simple": "What text can you read in this image?"
}

# Error Messages
ERROR_MESSAGES = {
    "api_key_missing": "Mistral API key is required",
    "api_key_invalid": "Invalid Mistral API key",
    "model_not_available": "Model not available",
    "analysis_failed": "Analysis failed",
    "no_detections": "No detections provided",
    "invalid_frame": "Invalid frame provided",
    "temp_file_error": "Temporary file error"
}

# File Extensions
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
TEMP_FILE_EXTENSION = '.jpg'

# Performance Settings
MAX_CONCURRENT_ANALYSES = 5
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
