from huggingface_hub import InferenceClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Load configuration from config.json
config = json.load(open("config.json"))

class HFModelLoader:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        # Initialize all models
        for model_name in config["models"].values():
            try:
                client = InferenceClient(
                    api_key=os.getenv("HF_TOKEN"),
                    model=model_name,
                    timeout=30
                )
                self.models[model_name] = client
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    def generate_response(
        self,
        model_name,
        prompt,
        max_length=512,
        temperature=0.5,
        top_p=None  # Add top_p parameter for sampling control
    ):
        try:
            # Get the appropriate client
            client = self.models.get(model_name)
            if not client:
                raise ValueError(f"Model {model_name} not found.")
            
            # Adjust parameters based on model requirements
            params = {
                "max_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p
            }

            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                **params
            )

            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating response for {model_name}: {e}")
            return str(e)

# Initialize the model loader
loader = HFModelLoader()

if __name__ == "__main__":
    # Example usage
    models = config["models"].values()
    
    for model in models:
        prompt = "What is the capital of France?"
        response = loader.generate_response(
            model,
            prompt,
            max_length=256,  # Adjust max_length as needed
            temperature=0.5,
            top_p=None      # Set to None for deterministic output
        )
        
        print(f"\nResponse from {model}:")
        print(response)