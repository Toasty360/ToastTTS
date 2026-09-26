"""Find out whether your speakers swallow the start of audio, and why.

Plays three short beeps (low, middle, high) in three different ways, with
a pause between tests so the output can go back to sleep. For each test,
count how many beeps you hear:

  A  device opened, beeps start immediately          (how most apps play)
  B  device opened, 1 s of pure digital silence, then beeps
  C  device opened with quiet room tone for 0.6 s, then beeps  (ToastTTS)

Reading the result:
  A cut, B fine          -> the device needs time after opening; any warm-up fixes it
  A and B cut, C fine    -> the output mutes on digital silence (common with HDMI/
                            DisplayPort monitor speakers, TVs, some Bluetooth); it
                            needs a quiet signal, not silence, to stay awake
  all three fine         -> the swallowing is elsewhere (tell us!)

  .venv\\Scripts\\python scripts\\audio_check.py
"""

import time

import numpy as np
import sounddevice as sd

from toast.player import DEVICE_WAKE_S, LivePlayer

SR = 22050
GAP_BETWEEN_TESTS_S = 4.0


def beeps():
    parts = []
    for freq in (440, 660, 880):
        t = np.arange(int(SR * 0.18)) / SR
        tone = 0.3 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
        parts += [tone, np.zeros(int(SR * 0.12))]
    return np.concatenate(parts).astype(np.float32)


def run_test(label, description, keep_awake, wait_s):
    print(f"\nTest {label}: {description}")
    time.sleep(1.0)
    player = LivePlayer(SR, keep_awake=keep_awake)
    time.sleep(wait_s)
    player.add(beeps())
    player.wait()
    player.close()
    time.sleep(GAP_BETWEEN_TESTS_S)


def main():
    print(f"Output device: {sd.query_devices(kind='output')['name']}")
    print("Count the beeps in each test (3 = nothing lost).")
    run_test("A", "beeps immediately after opening", keep_awake=False, wait_s=0.0)
    run_test("B", "1 s of digital silence first", keep_awake=False, wait_s=1.0)
    run_test("C", f"{DEVICE_WAKE_S} s of quiet room tone first (ToastTTS)", keep_awake=True, wait_s=DEVICE_WAKE_S)
    print("\nHow many beeps did you hear in A, B and C?")


if __name__ == "__main__":
    main()
