import pvporcupine
import pyaudio
import struct

class WakeWordDetector:
    def __init__(self, wake_word="jarvis"):
        self.wake_word = wake_word

    def listen(self):
        wake_word_detected = False
        porcupine = pvporcupine.create(access_key="", keywords=[self.wake_word])
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        print("Listening...")

        try:
            while not(wake_word_detected):
                pcm = stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")
                    wake_word_detected = True
        except KeyboardInterrupt:
            print("Interrupted by user.")
            return False
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()

        return wake_word_detected