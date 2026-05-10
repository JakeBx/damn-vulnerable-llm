from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

# Your fine-grained token with 'Inference' permissions
client = InferenceClient(
    provider="together", # Or other providers like 'groq', 'fal-ai'
    api_key=os.environ["HF_INFERENCE_TOKEN"] 
)

# This call will now draw from your HF prepaid credits/balance
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=500,
)

print(response.choices[0].message.content)