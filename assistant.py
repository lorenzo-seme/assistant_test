import time, subprocess
import sounddevice as sd
import numpy as np
import wave
from piper import PiperVoice
import whisper
import torch

# Percorsi modelli
piper_model_path = "models/it_IT-paola-medium.onnx"
piper_config_path = "models/it_IT-paola-medium.onnx.json"
whisper_model = "base"  # Puoi modificare a seconda del modello di Whisper che usi

# Funzione per registrare l'audio dal microfono
def record_audio(duration=10, sample_rate=16000):
    print("Inizio registrazione...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Attendere che la registrazione finisca
    print("Registrazione completata.")
    return audio_data

# Funzione per salvare l'audio in un file WAV
def save_wav(audio_data, filename="audio.wav", sample_rate=16000):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    print(f"Audio salvato come {filename}")

# Carica il modello di Piper
piper_voice = PiperVoice.load(piper_model_path, piper_config_path)

# Funzione per sintetizzare la risposta in audio
def synthesize_speech(text, filename="output.wav"):
    with wave.open(filename, "wb") as wf:
        piper_voice.synthesize(text, wf)
    print(f"Risposta sintetizzata come {filename}")

# Funzione per eseguire Whisper e ottenere il testo dall'audio
def transcribe_audio(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language="it")
    return result['text']

# Interazione con Llama (assumendo che sia già installato)
def generate_response_from_llama(input_text):
    # Esegui il comando di Llama per generare la risposta
    llama_command = f"ollama run llama3.2 \"{input_text}\""
    print(f"Chiamando Llama con il testo: {input_text}")
    result = subprocess.run(llama_command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        response = result.stdout.strip()
        print("Risposta da Llama:", response)
        return response
    else:
        print("Errore nella generazione della risposta con Llama")
        return "Errore durante la generazione della risposta."

# Funzione principale
def main():
    while True:
        # Registra l'audio
        audio_data = record_audio()
        save_wav(audio_data)

        # Trascrivi l'audio in testo
        transcribed_text = transcribe_audio("audio.wav")
        print(f"Testo trascritto: {transcribed_text}")

        # Passa il testo a Llama (implementa la risposta di Llama qui se necessario)
        response = generate_response_from_llama(transcribed_text)
        synthesize_speech(response, "response.wav")

        # Esegui il playback dell'audio sintetizzato
        sd.play(np.fromfile("response.wav", dtype=np.int16), 22050)
        sd.wait()

        time.sleep(2)  # Aggiungi una pausa prima della prossima registrazione

if __name__ == "__main__":
    main()
