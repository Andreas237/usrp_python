import uhd
import time
import numpy as np

def main():
    # Initialize USRP device
    usrp = uhd.usrp.MultiUSRP()

    # Configuration parameters
    sample_rate = 1e6  # 1 MSps
    center_freq_start = 70e6  # Start of frequency range (70 MHz)
    center_freq_end = 6e9  # End of frequency range (6 GHz)
    step_size = 10e6  # Frequency step size (10 MHz)
    gain = 40  # Gain in dB
    num_samples = int(sample_rate * 5)  # 5 seconds worth of samples

    # Configure Rx channel
    usrp.set_rx_rate(sample_rate)
    usrp.set_rx_gain(gain)

    # Sweep through frequencies
    freq = center_freq_start
    while freq <= center_freq_end:
        usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        print(f"Tuned to {freq/1e6} MHz")

        # Capture samples
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")  # Complex floats
        rx_stream = usrp.get_rx_stream(stream_args)
        buffer = np.zeros((num_samples,), dtype=np.complex64)

        rx_metadata = uhd.types.RXMetadata()
        samples_received = rx_stream.recv(buffer, rx_metadata, timeout=6.0)

        if samples_received > 0:
            print(f"Captured {samples_received} samples at {freq/1e6} MHz")

        # Save the captured samples (optional)
        filename = f"samples_{int(freq/1e6)}MHz.npy"
        np.save(filename, buffer[:samples_received])
        print(f"Saved samples to {filename}")

        # Move to the next frequency
        freq += step_size

    print("Frequency sweep completed.")

if __name__ == "__main__":
    main()
