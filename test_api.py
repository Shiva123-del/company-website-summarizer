
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env
load_dotenv()

# Connect to OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Send a test request
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Say hello to Shivam in one sentence."
)

# Display the response
print(response.output_text)