import google.generativeai as genai
import yaml
import os

# Load API key from your config file
try:
    with open("config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)
    api_key = config.get('gemini', {}).get('api_key')
    
    if not api_key:
        print("API key not found in config/settings.yaml")
        exit()
        
    genai.configure(api_key=api_key)

except FileNotFoundError:
    print("Error: config/settings.yaml not found. Make sure you are running this from your project root.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

print("--- Available Generative Models ---")
for m in genai.list_models():
  # We only care about models that can be used for generating text content
  if 'generateContent' in m.supported_generation_methods:
    print(f"Model Name: {m.name}\n  - Supported Methods: {m.supported_generation_methods}\n")