from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from scipy.signal import welch
import time
import uhd

samples_dir = './samples'
plots_dir = './plots'
clear_after_plotting = False

logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (func=%(funcName)s) (PID=%(process)d):%(message)s'
    )

def empty_dir(folder: str):
    cleared_files = 0
    statinfo_start = os.stat(folder).st_size
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            cleared_files += 1
        except Exception as e:
            print(f"Failed to remove {file_path} due to {e}")
    statinfo_end = os.stat(folder).st_size
    logging.info(f"Finished clearing {cleared_files} old files from {folder}, which saved {(statinfo_end - statinfo_start) // 1e6}Mb")



def plot_samples(file_path: str, sample_rate, center_freq, samples, NFFT):
    
    spectrogram_filename = plots_dir +"/" + f"spectrogram_{int(center_freq // 1e6)}MHz.png"
    
    try:
        # plt.specgram(samples, NFFT=NFFT, Fs=sample_rate, cmap='viridis')
        # plt.title(f"Spectrogram of {center_freq}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Frequency MHz")
        # plt.colorbar(label="Intensity (dB)")
        # plt.savefig(spectrogram_filename)
        # plt.close()
        
        freqs, psd = welch(samples, 
                            fs=sample_rate, 
                            window='blackmanharris', 
                            nperseg=1024, 
                            scaling='spectrum',
                            return_onesided=False)        
        logging.info(f'Created Welch PSD')
        # Create a plot for the PSD
        plt.figure(figsize=(8, 4))
        

        freq_axis = freqs + center_freq
        plt.plot(np.ravel(freq_axis)/ 1e6, np.sqrt(np.ravel(psd))) # ravel each due to different shapes

        plt.title(f"PSD at {center_freq/1e6:.2f} MHz")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("PSD (V^2/Hz)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(spectrogram_filename)
        logging.info(f'Save Welch PSD in {spectrogram_filename}')
        if clear_after_plotting:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            logging.info(f"Removed {file_path} ")
    except Exception as e:
        logging.error(f"Failed to remove {file_path} due to {e}")



def main():
    # Initialize USRP device
    usrp = uhd.usrp.MultiUSRP()

    # Configuration parameters
    duration_seconds = 5
    sample_rate = 30e6  # Msps
    center_freq_start = 70e6  # Start of frequency range (70 MHz)
    center_freq_end = 6e9  # End of frequency range (6 GHz)
    step_size = 10e6  # Frequency step size (10 MHz)
    gain = 5  # Gain in dB
    # num_samples = int(sample_rate * duration_seconds)  # 5 seconds worth of samples
    num_samples = 4096
    channels = [0] # use a single channel, must be as list

    # Configure Rx channel
    usrp.set_rx_rate(sample_rate)
    usrp.set_rx_gain(gain)

    # Sweep through frequencies
    freq = center_freq_start
    for freq in np.arange(center_freq_start, center_freq_end, step_size):
    # while freq <= center_freq_end:
        usrp.set_rx_freq(uhd.types.TuneRequest(freq))
        logging.info(f"Tuned to {freq/1e6} MHz")

        # Capture samples
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")  # Complex floats
        rx_stream = usrp.get_rx_stream(stream_args)
        buffer = np.zeros((num_samples,), dtype=np.complex64)

        rx_metadata = uhd.types.RXMetadata()
        samples_received = rx_stream.recv(buffer, rx_metadata, timeout=6.0)
        
        samps = usrp.recv_num_samps(num_samples, freq, sample_rate, channels, gain)
        
        filename = samples_dir + "/" +  f"samples_{int(freq/1e6)}MHz.npy"
        args = [(filename, sample_rate, freq, samps, 4096)]
        if samples_received > 0:
            logging.info(f"Captured {samples_received} samples at {freq/1e6} MHz")
            # Save the captured samples (optional)
            np.save(filename, buffer[:samples_received])
            logging.info(f"Saved samples to {filename}")
        elif len(samps) > 0:
            logging.info(f"Captured {samps.shape[1]} samples at {freq/1e6} MHz from UHD example")
            with open(filename, 'wb') as out_file:
                np.save(out_file, samps, allow_pickle=False, fix_imports=False)
            logging.info(f"Saved np samples to {filename}")
            results = Parallel(n_jobs=1)(delayed(plot_samples)(*a) for a in args)


        # Move to the next frequency
        freq += step_size

    logging.info("Frequency sweep completed.")

if __name__ == "__main__":
    
    empty_dir(samples_dir)
    empty_dir(plots_dir)
    main()
