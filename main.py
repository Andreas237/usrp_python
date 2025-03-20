"""
@file uhd_spectrogram.py
@brief This script captures radio frequency samples from a USRP device and generates spectrograms for each captured frequency.
It uses the `uhd` library for hardware control and interaction with the USRP, and it leverages `scipy` for signal processing tasks such as Welch's method for power spectral density estimation and spectrogram generation.
@author [Your Name]
@date [Date]
"""

from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from scipy.signal import welch, spectrogram
import time
import uhd



# Define directories for samples and plots
samples_dir = './samples'
plots_dir = './plots'
clear_after_plotting = True # change this to `False` if you want to save the samples from each sweep



# Configure logging to print info messages with function name, process ID, etc.
logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (func=%(funcName)s) (PID=%(process)d) (processName=%(processName)s):%(message)s'
    )



def empty_dir(folder: str):
    """
    Clears the specified directory of all files and subdirectories.
    
    @param folder: The path to the directory that needs to be cleared.
    """
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
            logging.warn(f"Failed to remove {file_path} due to {e}")
    statinfo_end = os.stat(folder).st_size
    logging.info(f"Finished clearing {cleared_files} old files from {folder}, which saved {(statinfo_end - statinfo_start) // 1e6}Mb")



def plot_samples(file_path: str, sample_rate, center_freq, samples, NFFT):
    """
    Plots the captured samples as a spectrogram.
    
    @param file_path: The path to the saved numpy array of samples.
    @param sample_rate: The sampling rate of the USRP device.
    @param center_freq: The central frequency at which the USRP is tuned.
    @param samples: The captured radio frequency samples.
    @param NFFT: Number of points in each FFT.
    """

    # Define the output file path for the spectrogram image
    spectrogram_filename = plots_dir +"/" + f"spectrogram_{int(center_freq // 1e6)}MHz.png"
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Plot the Power Spectral Density (PSD) using Welch's method
        freqs, psd = welch(samples, 
                            fs=sample_rate, 
                            window='blackmanharris', 
                            nperseg=1024, 
                            scaling='spectrum',
                            return_onesided=False)        
        logging.info(f'Created Welch PSD')

        freq_axis = freqs + center_freq
        ax1.plot(np.ravel(freq_axis)/ 1e6, np.sqrt(np.ravel(psd))) # ravel each due to different shapes
        ax1.set_title(f"PSD at {center_freq/1e6:.2f} MHz")
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("PSD (V^2/Hz)")
        ax1.grid()
        
        
        f, t, Sxx = spectrogram(samples, 
                                sample_rate,
                                return_onesided=False)
        logging.info(f'Created Spectrogram')
        freq_axis2 = f + center_freq

        ax2.pcolormesh(freq_axis2 / 1e6, t, np.squeeze(Sxx).T, shading='auto') # Sxx has 3 dims, but needs only the two matching f and t.
                                                                   # Also, plot frequency on the x-axis with the transpose
        ax2.set_title(f"PSD at {center_freq/1e6:.2f} MHz")
        ax2.set_ylabel("Time (s)")
        ax2.set_xlabel("Frequency MHz")
        ax2.grid()
        plt.tight_layout()
        fig.savefig(spectrogram_filename)
        if clear_after_plotting:
            # Remove the file after plotting
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            logging.info(f"Removed {file_path} ")
    except IOError as e:
        logging.error(f"Failed to remove {file_path} due to {e}")
    except TypeError as e:
        logging.error(f"Failed to create a spectrogram {spectrogram_filename} from {file_path} due to {e}")
    except Exception as e:
        logging.error(f'f dims: {f.shape}\tt dims {t.shape}\t Sxx dims {Sxx.shape}')
        raise Exception

# Main function to configure and run the USRP for frequency sweeping
def main():
    """
    Main function to configure and run the USRP for frequency sweeping, capture samples, and generate spectrograms.
    """

    # Initialize USRP device
    usrp = uhd.usrp.MultiUSRP()

    # Configuration parameters # TODO: use argparser to clean this out
    duration_seconds = 5
    sample_rate = 30e6  # Msps
    center_freq_start = 70e6  # Start of frequency range (70 MHz)
    center_freq_end = 6e9  # End of frequency range (6 GHz)
    step_size = np.floor(sample_rate / 2)

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
        try:
            samps = usrp.recv_num_samps(num_samples, freq, sample_rate, channels, gain)
        except RuntimeError as e:
            logging.error(f"Caught an RuntimeError exception when sampling the receiver.  Sleeping for 10 seconds and trying again.  Here's the exception: {e}")
            time.sleep(10)
            samps = usrp.recv_num_samps(num_samples, freq, sample_rate, channels, gain)
        
        # Save sample; perhaps we want to save the array of samples.
        filename = samples_dir + "/" +  f"samples_{int(freq/1e6)}MHz.npy"
        args = [(filename, sample_rate, freq, samps, 4096)]
        
        logging.info(f"Captured {samps.shape[1]} samples at {freq/1e6} MHz from UHD example")
        with open(filename, 'wb') as out_file:
            np.save(out_file, samps, allow_pickle=False, fix_imports=False)
        logging.info(f"Saved np samples to {filename}")

        # Make PSD and Spectrogram of the captured samples
        results = Parallel(n_jobs=1)(delayed(plot_samples)(*a) for a in args)

        # Move to the next frequency
        freq += step_size

    logging.info("Frequency sweep completed.")

if __name__ == "__main__":
    
    empty_dir(samples_dir)
    empty_dir(plots_dir)
    main()
