import whisper

class Transcriber:
    """
        A class for transcribing speech from audio files using OpenAI's Whisper model.

        This class loads a Whisper model and provides functionality to convert
        spoken audio into written text in a specified language.

        Attributes:
            model_size (str): The size of the Whisper model to load (e.g., "tiny", "base", "small", "medium", "large").
            language (str): The language code for transcription (e.g., "it" for Italian, "en" for English).

        Methods:
            transcribe(audio_file: str) -> str:
                Transcribes the given audio file and returns the recognized text.
    """
    def __init__(self, model_size="small", language="it"):
        """
            Initializes the Transcriber with the specified model size and language.

            Args:
                model_size (str): The size of the Whisper model to use.
                language (str): The language to use for transcription.
        """
        self.model_size = model_size
        self.language = language

    def transcribe(self, audio_file):
        """
            Transcribes the audio content of a file to text using Whisper.

            Args:
                audio_file (str): Path to the audio file to transcribe.

            Returns:
                str: The transcribed text.
        """
        model = whisper.load_model(self.model_size)
        result = model.transcribe(audio_file, language=self.language)
        return result['text']