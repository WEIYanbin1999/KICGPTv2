"""
API Configuration Helper for KICGPTv2
Supports Qwen-2.5-72B via OpenAI-compatible API endpoint
"""
import openai
import os

def configure_openai_api(api_key=None, api_base=None):
    """
    Configure OpenAI API for KICGPTv2
    
    Args:
        api_key: API key for authentication
        api_base: API endpoint base URL
    
    As per reviewer requirement:
        - Model: Qwen/Qwen2.5-72B-Instruct (accessible via API)
        - API Base: https://api.pumpkinaigc.online/v1
    """
    # Set API key
    if api_key:
        openai.api_key = api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("API key must be provided via argument or OPENAI_API_KEY environment variable")
    
    # Set API base for Qwen-2.5-72B
    if api_base:
        openai.api_base = api_base
    elif os.getenv("OPENAI_API_BASE"):
        openai.api_base = os.getenv("OPENAI_API_BASE")
    else:
        # Default to Qwen-2.5-72B endpoint as specified by user
        openai.api_base = 'https://api.pumpkinaigc.online/v1'
    
    print(f"API configured - Base: {openai.api_base}")
    return openai.api_base


def get_model_name(model=None):
    """
    Get the model name for API calls
    
    Default: Qwen/Qwen2.5-72B-Instruct as specified in reviewer requirements
    """
    if model:
        return model
    elif os.getenv("OPENAI_MODEL"):
        return os.getenv("OPENAI_MODEL")
    else:
        # Use Qwen-2.5-72B as default (can be accessed via API)
        return "Qwen/Qwen2.5-72B-Instruct"
