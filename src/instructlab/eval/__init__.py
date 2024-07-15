# Standard
import os

# Inherit logging from caller rather than from vLLM
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
