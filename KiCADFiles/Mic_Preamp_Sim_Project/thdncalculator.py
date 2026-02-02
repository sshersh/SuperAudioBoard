# from https://gist.github.com/endolith/246092

from __future__ import division
import sys
from scipy.signal.windows import blackmanharris
from scipy import signal as scipy_signal
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, arange, log10
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(mean(absolute(a)**2))


def find_range(f, x):
    """
    Find range between nearest local minima from peak at index x
    """
    uppermin = len(f) - 1
    lowermin = 0
    
    for i in arange(x+1, len(f)-1):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)


def THDN(signal, sample_rate):
    """
    Measure the THD+N for a signal and return the results

    Returns the estimated fundamental frequency, the measured THD+N,
    the full FFT, and frequency bins.  This is calculated from the ratio 
    of the entire signal before and after notch-filtering.

    Currently this tries to find the "skirt" around the fundamental and notch
    out the entire thing.  A fixed-width filter would probably be just as good,
    if not better.
    """
    # Get rid of DC and window the signal

    # TODO: Do this in the frequency domain, and take any skirts with it?
    signal -= mean(signal)
    windowed = signal * blackmanharris(len(signal))  # TODO Kaiser?

    # Measure the total signal before filtering but after windowing
    total_rms = rms_flat(windowed)

    # Find the peak of the frequency spectrum (fundamental frequency), and
    # filter the signal by throwing away values between the nearest local
    # minima
    f = rfft(windowed)
    f_original = f.copy()  # Keep original FFT for plotting
    i = argmax(abs(f))

    # Calculate frequency
    frequency = sample_rate * (i / len(windowed))
    lowermin, uppermin = find_range(abs(f), i)
    f[lowermin: uppermin] = 0

    # Transform noise back into the signal domain and measure it
    # TODO: Could probably calculate the RMS directly in the frequency domain
    # instead
    noise = irfft(f)
    THDN = rms_flat(noise) / total_rms
    
    # Generate frequency bins
    freq_bins = np.fft.rfftfreq(len(windowed), 1/sample_rate)
    
    return frequency, THDN, f_original, freq_bins

def load(filename):
    """
    Load a csv
    """
    with open(filename, 'r') as f:
        header = f.readline().split(';')[:-1]
        time = []
        signals = [[] for _ in range(len(header) -1)]
        for line in f:
            values = line.split(';')[:-1]
            time.append(float(values[0]))
            for i, value in enumerate(values[1:]):
                signals[i].append(float(value))

    sample_rate = 1 / (time[1] - time[0])
    signal = np.array(signals).transpose()
    print('Loaded "%s": %d channels, %d samples, %.1f Hz' % (filename, signal.shape[1], signal.shape[0], sample_rate))
    
    print(signal)

    return signal, sample_rate, signal.shape[1], header[1:]


def analyze_channels(filename, function):
    """
    Given a filename, run the given analyzer function on each channel of the
    file and output results as a table
    """
    signal, sample_rate, channels, header = load(filename)
    print('Analyzing "' + filename + '"...')
    print()
    
    # Collect results for all channels
    results = []
    fft_data = []
    for ch_no, channel in enumerate(signal.transpose()):
        freq, thdn, fft, freq_bins = function(channel, sample_rate)
        results.append((ch_no + 1, freq, thdn))
        fft_data.append((fft, freq_bins))
    
    # Print table header
    print("Signal | Frequency (Hz) | THD+N (%)  | THD+N (dB)")
    print("-------|----------------|------------|------------")
    
    # Print each row
    for signal_num, freq, thdn in results:
        thdn_percent = thdn * 100
        thdn_db = 20 * log10(thdn)
        print("%-6s | %14.2f | %10.4f | %10.1f" % (header[signal_num - 1], freq, thdn_percent, thdn_db))
    
    # Plot spectrum for all signals
    print("\nGenerating spectrum plot...")
    plot_spectrum(fft_data, header, filename)

def plot_spectrum(fft_data, header, filename):
    """
    Plot the frequency spectrum of each signal separately side by side
    """
    num_signals = len(fft_data)
    
    # Create subplots side by side
    fig, axes = plt.subplots(1, num_signals, figsize=(6 * num_signals, 6))
    
    # Handle single subplot case
    if num_signals == 1:
        axes = [axes]
    
    for idx, ((fft, freq_bins), label) in enumerate(zip(fft_data, header)):
        ax = axes[idx]
        
        # Calculate magnitude in dB
        magnitude_db = 20 * log10(np.abs(fft) + 1e-10)
        
        # Plot spectrum
        ax.plot(freq_bins, magnitude_db, alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits and ticks in 1k increments
        max_freq = min(freq_bins[-1], 20000)
        ax.set_xlim([0, max_freq])
        x_ticks = np.arange(0, max_freq + 1000, 1000)
        x_labels = [f'{int(x/1000)}k' if x > 0 else '0' for x in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # Set y-axis ticks in 10dB increments
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min / 10) * 10, np.ceil(y_max / 10) * 10 + 10, 10)
        ax.set_yticks(y_ticks)
    
    fig.suptitle(f'Frequency Spectrum - {filename}', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_filename = filename.rsplit('.', 1)[0] + '_spectrum.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Spectrum saved to {output_filename}")
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    files = sys.argv[1:]
    if files:
        for filename in files:
            try:
                analyze_channels(filename, THDN)
            except Exception as e:
                print('Couldn\'t analyze "' + filename + '"')
                print(e)
            print()
    else:
        sys.exit("You must provide at least one file to analyze")
