"""Test PyAudioWPatch loopback detection."""
import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
print(f"Device count: {p.get_device_count()}")
print("\n=== Loopback Devices ===")
found = 0
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d.get("isLoopbackDevice", False):
        found += 1
        print(f"  [{d['index']}] {d['name']} ({d['maxInputChannels']}ch, {d['defaultSampleRate']}Hz)")

if not found:
    print("  (none found)")

# try opening the first loopback device briefly
if found:
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    print(f"\nDefault output: {default_out['name']}")

    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("isLoopbackDevice", False):
            print(f"\nTesting loopback [{d['index']}] {d['name']}...")
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=d["maxInputChannels"],
                    rate=int(d["defaultSampleRate"]),
                    input=True,
                    input_device_index=d["index"],
                    frames_per_buffer=512,
                )
                data = stream.read(512, exception_on_overflow=False)
                stream.stop_stream()
                stream.close()
                print(f"  SUCCESS — got {len(data)} bytes")
            except Exception as e:
                print(f"  FAILED: {e}")
            break

p.terminate()
print("\nDone!")
