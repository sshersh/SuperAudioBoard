#!/usr/bin/env python3
"""
Script to sweep Rg resistance and capture differential voltages using ngspice
"""

import subprocess
import os
import re
import csv
import numpy as np
import argparse

def modify_netlist(original_file, output_file, rg_value):
    """
    Modify the netlist to change Rg (R6) value and add output commands
    """
    with open(original_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            # Replace Rg (R6) value
            if line.startswith('R6 '):
                parts = line.split()
                # R6 Net-_R6-Pad1_ Net-_C4-Pad2_ 5k
                f.write(f'{parts[0]} {parts[1]} {parts[2]} {rg_value}\n')
            # Replace .end with control commands
            elif line.strip() == '.end':
                # Add control section before .end
                f.write('\n.control\n')
                f.write('run\n')
                f.write('set wr_singlescale\n')
                f.write('set wr_vecnames\n')
                f.write('option numdgt=7\n')
                f.write('wrdata output.csv time v(/OUT_P) v(/OUT_N) v(/IN_P) v(/IN_N)\n')
                f.write('quit\n')
                f.write('.endc\n')
                f.write('\n.end\n')
            else:
                f.write(line)

def run_ngspice(netlist_file):
    """
    Run ngspice with the given netlist and return the output file path
    """
    # Run ngspice in batch mode
    result = subprocess.run(
        ['ngspice', '-b', netlist_file],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(netlist_file))
    )
    
    if result.returncode != 0:
        print(f"Error running ngspice: {result.stderr}")
        return None
    
    return 'output.csv'

def parse_ngspice_output(output_file):
    """
    Parse ngspice output CSV and return time and voltage arrays
    """
    time = []
    v_out_p = []
    v_out_n = []
    v_in_p = []
    v_in_n = []
    
    with open(output_file, 'r') as f:
        # Skip header lines until we find the data
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('Index'):
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    time.append(float(parts[0]))
                    v_out_p.append(float(parts[1]))
                    v_out_n.append(float(parts[2]))
                    v_in_p.append(float(parts[3]))
                    v_in_n.append(float(parts[4]))
                except ValueError:
                    continue
    
    return (np.array(time), np.array(v_out_p), np.array(v_out_n), 
            np.array(v_in_p), np.array(v_in_n))

def sweep_rg(netlist_file, rg_values, output_csv):
    """
    Sweep Rg values and store results in CSV
    """
    temp_netlist = 'temp_netlist.cir'
    all_data = {}
    time_data = None
    
    print(f"Sweeping Rg values: {rg_values}")
    
    for rg_val in rg_values:
        print(f"Running simulation with Rg = {rg_val}...")
        
        # Modify netlist with current Rg value
        modify_netlist(netlist_file, temp_netlist, rg_val)
        
        # Run ngspice
        output_file = run_ngspice(temp_netlist)
        
        if output_file and os.path.exists(output_file):
            # Parse results
            time, v_out_p, v_out_n, v_in_p, v_in_n = parse_ngspice_output(output_file)
            
            if time_data is None:
                time_data = time
            
            # Calculate differential voltages
            v_out_diff = v_out_p - v_out_n
            v_in_diff = v_in_p - v_in_n
            
            # Store in dictionary
            all_data[rg_val] = {
                'v_out_diff': v_out_diff,
                'v_in_diff': v_in_diff
            }
            
            print(f"  Completed: {len(time)} samples")
        else:
            print(f"  Failed to generate output")
    
    # Write combined CSV
    if time_data is not None and all_data:
        print(f"\nWriting results to {output_csv}...")
        with open(output_csv, 'w', newline='') as f:
            # Create header
            header = ['time']
            for rg_val in rg_values:
                if rg_val in all_data:
                    header.append(f'V_OUT_DIFF_Rg_{rg_val}')
            header.append('V_IN_DIFF')
            
            writer = csv.writer(f, delimiter=';')
            writer.writerow([h + ';' for h in header])
            
            # Write data rows
            for i in range(len(time_data)):
                row = [time_data[i]]
                for rg_val in rg_values:
                    if rg_val in all_data:
                        row.append(all_data[rg_val]['v_out_diff'][i])
                # Add input differential (same for all Rg values, just use last one)
                if rg_values[-1] in all_data:
                    row.append(all_data[rg_values[-1]]['v_in_diff'][i])
                
                writer.writerow([str(x) + ';' for x in row])
        
        print(f"Done! Results saved to {output_csv}")
    
    # Cleanup
    if os.path.exists(temp_netlist):
        os.remove(temp_netlist)
    if os.path.exists('output.csv'):
        os.remove('output.csv')

def format_resistance(value):
    """Format resistance value with appropriate SI suffix"""
    if value >= 1e6:
        return f"{value/1e6:.0f}Meg"
    elif value >= 1e3:
        return f"{value/1e3:.0f}k"
    else:
        return f"{value:.0f}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep Rg resistance and capture differential voltages')
    parser.add_argument('start', type=float, help='Start resistance value (in ohms)')
    parser.add_argument('stop', type=float, help='Stop resistance value (in ohms)')
    parser.add_argument('--points', type=int, help='Number of points in sweep', default=10)
    parser.add_argument('--rule', choices=['linear', 'log'], default='log', 
                        help='Sweep rule: linear or log (default: log)')
    parser.add_argument('--netlist', default='first_stage_netlist.txt', 
                        help='Input netlist file (default: first_stage_netlist.txt)')
    parser.add_argument('--output', default='rg_sweep_results.csv', 
                        help='Output CSV file (default: rg_sweep_results.csv)')
    
    args = parser.parse_args()
    
    # Generate Rg values based on sweep rule
    if args.rule == 'linear':
        rg_values_numeric = np.linspace(args.start, args.stop, args.points)
    else:  # log
        rg_values_numeric = np.logspace(np.log10(args.start), np.log10(args.stop), args.points)
    
    # Format with SI suffixes
    rg_values = [format_resistance(val) for val in rg_values_numeric]
    
    print(f"Sweep configuration:")
    print(f"  Start: {args.start} ohms ({rg_values[0]})")
    print(f"  Stop: {args.stop} ohms ({rg_values[-1]})")
    print(f"  Points: {args.points}")
    print(f"  Rule: {args.rule}")
    print()
    
    # Run the sweep
    sweep_rg(args.netlist, rg_values, args.output)
