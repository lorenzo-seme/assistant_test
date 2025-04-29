import wave, os
from piper import PiperVoice

# Percorso al tuo modello e al file di configurazione
model_path = "../models/it_IT-paola-medium.onnx"
config_path = "../models/it_IT-paola-medium.onnx.json"

print(os.path.exists(model_path))  # Dovrebbe restituire True
print(os.path.exists(config_path))

# Carica il modello
voice = PiperVoice.load(model_path, config_path)

# Apri un file WAV per scrivere l'audio sintetizzato
with wave.open("output.wav", "wb") as wav_file:
    # Sintetizza il testo in un file WAV
    voice.synthesize("Ciao, questa è la mia voce. Come posso aiutarti?", wav_file)
