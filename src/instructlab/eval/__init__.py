# Standard
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    os.environ["OPENAI_API_KEY"] = "NO_API_KEY"
