import os
import wave

import sounddevice as sd

class Recorder:
    """
        A class for recording audio from the microphone and saving it to a WAV file.

        This class uses the `sounddevice` library to capture audio input and stores it
        in 16-bit PCM format using the `wave` module.

        Attributes:
            sample_rate (int): The sampling rate in Hz (default: 16000).
            max_duration (int): Maximum recording duration in seconds (default: 10).

        Methods:
            rec(filename="audio.wav"):
                Records audio from the microphone and saves it to a WAV file.

            save_rec(audio_data, filename):
                Saves the recorded audio data to a WAV file.
    """
    def __init__(self, sample_rate=16000, max_duration=10):
        """
            Initializes the Recorder with the given sample rate and maximum duration.

            Args:
                sample_rate (int): Sampling rate in Hz.
                max_duration (int): Maximum duration for the recording in seconds.
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def rec(self, filename="audio.wav"):
        """
           Records audio from the default input device and saves it as a WAV file.

           Args:
               filename (str): The output filename (default: "audio.wav").
        """
        print("Recording...")
        audio_data = sd.rec(int(self.max_duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording completed.")
        self.save_rec(audio_data, filename)

    def save_rec(self, audio_data, filename):
        """
                Saves the recorded audio data to a WAV file in 16-bit PCM format.

                Args:
                    audio_data: NumPy array containing the recorded audio.
                    filename (str): The filename where the audio will be saved.
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        print(f"Saved as {filename}")


