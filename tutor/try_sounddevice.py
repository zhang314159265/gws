"""
The tutorial is based on https://stackoverflow.com/questions/35344649/reading-input-sound-signal-using-python
"""

import sounddevice as sd
import soundfile as sf

fs = 44100
duration = 5  # seconds

myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype="float64")

print("Recording audio")
sd.wait()
print("Audio recording complete, play audio")
sd.play(myrecording, fs)
sd.wait()
print("Play audio complete")

path = "/tmp/my.wav"
sf.write(path, myrecording, fs, subtype="DOUBLE")
print(f"Auto written to {path}")
