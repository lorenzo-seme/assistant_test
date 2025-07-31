import wave

from piper import PiperVoice

class Synthesizer:
    """
        A class for speech synthesis using a Piper voice model.

        This class loads a Piper TTS (text-to-speech) model and allows converting
        input text into a spoken audio file in WAV format.

        Attributes:
            piper_voice (PiperVoice): An instance of the loaded Piper voice model.

        Methods:
            synthesize(text: str, filename: str = "response.wav") -> None:
                Converts input text to speech and saves it as a WAV file.
    """
    def __init__(self, piper_model_path, piper_config_path):
        """
            Initializes the Synthesizer by loading the specified Piper model.

            Args:
                piper_model_path (str): Path to the `.onnx` voice model file.
                piper_config_path (str): Path to the `.json` configuration file for the model.
        """

        self.piper_voice = PiperVoice.load(piper_model_path, piper_config_path)

    def synthesize(self, text, filename="response.wav"):
        """
            Synthesizes speech from input text and saves it as a WAV file.

            Args:
                text (str): The text to convert into speech.
                filename (str): The name of the output WAV file (default: "response.wav").

            The generated audio has the following properties:
                - Mono (1 channel)
                - 16-bit PCM (2 bytes per sample)
                - 22050 Hz sample rate (typical for Piper)
        """

        audio_chunks = self.piper_voice.synthesize(text)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(22050)  # Frequenza tipica per Piper (verifica dal tuo config)

            for chunk in audio_chunks:
                wf.writeframes(chunk.audio_int16_bytes)

