from assistant.generator import Generator
from assistant.synthesizer import Synthesizer

generator = Generator(model="phi3:3.8b-mini-4k-instruct-q4_K_M")
response = generator.generate("Il corriere ha messo che è passato e non c'era nessuno in casa ma non è vero, cosa faccio?")

piper_model_path = "../models/it_IT-paola-medium.onnx"
piper_config_path = "../models/it_IT-paola-medium.onnx.json"
synthesizer = Synthesizer(piper_model_path, piper_config_path)
synthesizer.synthesize(text=response, filename="test_response.wav")
print("Response synthesized")
