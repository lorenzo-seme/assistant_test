import os
import time

import numpy as np
import sounddevice as sd
from assistant.generator import Generator
from assistant.recorder import Recorder
from assistant.synthesizer import Synthesizer
from assistant.transcriber import Transcriber

# Voice model path
piper_model_path = "models/it_IT-paola-medium.onnx"
piper_config_path = "models/it_IT-paola-medium.onnx.json"

# Llama model
ollama_model = "smollm:135m"

if __name__ == "__main__":

    recorder = Recorder()
    transcriber = Transcriber()
    synthesizer = Synthesizer(piper_model_path, piper_config_path)
    generator = Generator(model=ollama_model)

    while True: # TODO: implement wake word

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
        sd.play(np.fromfile(synthesized_filename, dtype=np.int16), 22050)
        sd.wait()

        # Wait
        time.sleep(2)
