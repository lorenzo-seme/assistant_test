import os
import time

import numpy as np
import sounddevice as sd
from assistant.generator import Generator
from assistant.recorder import Recorder
from assistant.synthesizer import Synthesizer
from assistant.transcriber import Transcriber
from assistant.wake_word_detector import WakeWordDetector

# Voice model path
piper_model_path = "models/en_GB-alan-low.onnx"
piper_config_path = "models/en_GB-alan-low.onnx.json"


# Llama model
ollama_model = "tinyllama"

if __name__ == "__main__":
    waker = WakeWordDetector()
    recorder = Recorder(max_duration=5)
    transcriber = Transcriber()
    synthesizer = Synthesizer(piper_model_path, piper_config_path)
    generator = Generator(model=ollama_model)

    while True: # TODO: implement wake word
        if not waker.listen():
            continue
        print("Assistant activated!")

        # Recorder
        recorded_filename = os.path.join("audio", "input", "audio.wav") # TODO: aggiungi timestamp al filename
        recorder.rec(recorded_filename)

        # Transcriber - Whisper
        transcribed_text = transcriber.transcribe(recorded_filename)
        print(f"Transcribed text: {transcribed_text}")

        # Generator - Generate response with ollama
        response = generator.generate(transcribed_text)

        # Synthesizer - Piper
        synthesized_filename = os.path.join("audio", "output", "response.wav")
        synthesizer.synthesize(text=response, filename=synthesized_filename)
        print("Response synthesized")

        # Play synthesized audio file
        sd.play(np.fromfile(synthesized_filename, dtype=np.int16), 16000)
        sd.wait()

        # Wait
        time.sleep(2)
