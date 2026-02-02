from thdncalculator import load, THD_Discrete
import sys

signal_path = sys.argv[1] if len(sys.argv) > 1 else None
fundamental = sys.argv[2] if len(sys.argv) > 2 else None
num_averages = sys.argv[3] if len(sys.argv) > 3 else None

if signal_path:
    signal, sample_rate, _, _ = load(signal_path)
    THD_Discrete(signal[:,0], sample_rate, float(fundamental), int(num_averages))