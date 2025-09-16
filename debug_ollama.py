import requests

def debug_ollama():
    """Debug Ollama connection and model status"""
    try:
        # Check basic connection
        health_check = requests.get('http://localhost:11434/api/health', timeout=5)
        print("\n=== Ollama Health Check ===")
        print(f"Health endpoint status: {health_check.status_code}")
        
        # Check available models
        models_check = requests.get('http://localhost:11434/api/tags', timeout=5)
        print("\n=== Available Models ===")
        if models_check.status_code == 200:
            models = models_check.json()
            for model in models['models']:
                print(f"- {model['name']}")
        else:
            print("Could not fetch models:", models_check.status_code)
            
        # Test simple generation
        print("\n=== Testing Generation ===")
        test_prompt = {
            "model": "qwen:0.5b",
            "prompt": "Say hello",
            "stream": False
        }
        gen_check = requests.post('http://localhost:11434/api/generate', 
                                json=test_prompt, 
                                timeout=15)
        print(f"Generation status: {gen_check.status_code}")
        if gen_check.status_code == 200:
            print("Generation successful!")
            print("Response:", gen_check.json().get('response', ''))
        else:
            print("Generation failed!")
            print("Error:", gen_check.text)
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama server")
    except Exception as e:
        print(f"Error during debug: {str(e)}")

if __name__ == "__main__":
    print("Debugging Ollama connection...")
    debug_ollama()