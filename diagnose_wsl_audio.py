import sounddevice as sd

print("PortAudio devices visible from Python:\n")
print(sd.query_devices())
print("\nDefault input/output device:")
print(sd.default.device)
