"""
Example usage of IEEE P370 Time Domain Quality Check
"""

import numpy as np
import skrf as rf
from ieee_p370_quality_time_domain import quality_check

# Example 1: Load S-parameters from a touchstone file
def example_from_file():
    """Load S-parameters from a file and calculate time-domain metrics"""
    
    # Load a 2-port S-parameter file
    network = rf.Network('example.s2p')
    
    # Extract frequency and S-parameter data
    freq = network.f  # Frequency in Hz
    sdata = network.s  # S-parameters (freq x ports x ports)
    
    # Transpose to match IEEE P370 convention (ports x ports x freq)
    sdata_transposed = np.transpose(sdata, (1, 2, 0))
    
    # Set parameters
    port_num = network.nports
    data_rate = 25.0          # 25 Gbps
    sample_per_ui = 64        # 64 samples per unit interval
    rise_per = 0.35           # 35% rise time
    pulse_shape = 1           # 1=Gaussian, 2=First-order, 3=Gaussian filtered
    extrapolation_method = 1  # 1=Phase-based, 0=Zero-padding
    
    # Calculate time-domain quality metrics
    causality_mv, reciprocity_mv, passivity_mv = quality_check(
        freq=freq,
        sdata=sdata_transposed,
        port_num=port_num,
        data_rate=data_rate,
        sample_per_ui=sample_per_ui,
        rise_per=rise_per,
        pulse_shape=pulse_shape,
        extrapolation_method=extrapolation_method
    )
    
    print(f"IEEE P370 Time Domain Quality Metrics for {network.name}:")
    print(f"  Causality:   {causality_mv:.1f} mV")
    print(f"  Reciprocity: {reciprocity_mv:.1f} mV")
    print(f"  Passivity:   {passivity_mv:.1f} mV")
    print()
    
    # Interpret results
    print("Interpretation:")
    if causality_mv < 1.0:
        print("  ✓ Excellent causality (< 1 mV)")
    elif causality_mv < 5.0:
        print("  ✓ Good causality (< 5 mV)")
    else:
        print("  ⚠ Poor causality (≥ 5 mV)")
    
    return causality_mv, reciprocity_mv, passivity_mv


# Example 2: Create synthetic S-parameters for testing
def example_synthetic():
    """Create synthetic S-parameters and calculate metrics"""
    
    # Create frequency array (0-40 GHz)
    freq = np.linspace(0, 40e9, 2001)
    port_num = 2
    
    # Create synthetic S-parameters (2-port transmission line)
    sdata = np.zeros((port_num, port_num, len(freq)), dtype=complex)
    
    # Transmission line parameters
    loss_db_per_ghz = 0.5  # Loss in dB/GHz
    delay_ps = 100          # Delay in picoseconds
    z0 = 50                 # Characteristic impedance
    
    for i, f in enumerate(freq):
        if f == 0:
            # DC values
            sdata[0, 1, i] = 1.0
            sdata[1, 0, i] = 1.0
            sdata[0, 0, i] = 0.0
            sdata[1, 1, i] = 0.0
        else:
            # Frequency-dependent loss and phase
            loss_linear = 10**(-loss_db_per_ghz * f/1e9 / 20)
            phase = -2 * np.pi * f * delay_ps * 1e-12
            
            # S21 and S12 (transmission)
            sdata[0, 1, i] = loss_linear * np.exp(1j * phase)
            sdata[1, 0, i] = loss_linear * np.exp(1j * phase)
            
            # S11 and S22 (reflection) - small mismatch
            sdata[0, 0, i] = 0.1 * np.exp(1j * phase * 0.5)
            sdata[1, 1, i] = 0.1 * np.exp(1j * phase * 0.5)
    
    # Calculate metrics for different data rates
    data_rates = [10, 25, 56]  # Gbps
    
    print("Synthetic Transmission Line S-parameters:")
    print(f"  Loss: {loss_db_per_ghz} dB/GHz")
    print(f"  Delay: {delay_ps} ps")
    print()
    
    for data_rate in data_rates:
        causality_mv, reciprocity_mv, passivity_mv = quality_check(
            freq=freq,
            sdata=sdata,
            port_num=port_num,
            data_rate=data_rate,
            sample_per_ui=64,
            rise_per=0.35,
            pulse_shape=1,
            extrapolation_method=1
        )
        
        print(f"Data Rate: {data_rate} Gbps")
        print(f"  Causality:   {causality_mv:.1f} mV")
        print(f"  Reciprocity: {reciprocity_mv:.1f} mV")
        print(f"  Passivity:   {passivity_mv:.1f} mV")
        print()


# Example 3: Batch processing multiple files
def example_batch_processing():
    """Process multiple S-parameter files"""
    
    import os
    import pandas as pd
    
    # List of files to process
    file_list = [
        'channel1.s2p',
        'channel2.s2p',
        'channel3.s4p',
        # Add more files as needed
    ]
    
    # Parameters
    data_rate = 25.0
    sample_per_ui = 64
    rise_per = 0.35
    pulse_shape = 1
    extrapolation_method = 1
    
    results = []
    
    for filename in file_list:
        if os.path.exists(filename):
            try:
                # Load network
                network = rf.Network(filename)
                freq = network.f
                sdata = np.transpose(network.s, (1, 2, 0))
                
                # Calculate metrics
                causality_mv, reciprocity_mv, passivity_mv = quality_check(
                    freq=freq,
                    sdata=sdata,
                    port_num=network.nports,
                    data_rate=data_rate,
                    sample_per_ui=sample_per_ui,
                    rise_per=rise_per,
                    pulse_shape=pulse_shape,
                    extrapolation_method=extrapolation_method
                )
                
                results.append({
                    'Filename': filename,
                    'Ports': network.nports,
                    'Causality (mV)': causality_mv,
                    'Reciprocity (mV)': reciprocity_mv,
                    'Passivity (mV)': passivity_mv
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Create DataFrame and display results
    df = pd.DataFrame(results)
    print("\nBatch Processing Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('ieee_p370_time_domain_results.csv', index=False)
    print("\nResults saved to ieee_p370_time_domain_results.csv")


# Example 4: Compare different pulse shapes
def example_pulse_shapes():
    """Compare metrics using different pulse shapes"""
    
    # Load example network
    network = rf.Network('example.s2p')
    freq = network.f
    sdata = np.transpose(network.s, (1, 2, 0))
    
    # Common parameters
    params = {
        'freq': freq,
        'sdata': sdata,
        'port_num': network.nports,
        'data_rate': 25.0,
        'sample_per_ui': 64,
        'rise_per': 0.35,
        'extrapolation_method': 1
    }
    
    pulse_shapes = {
        1: "Gaussian",
        2: "First-order filter",
        3: "Gaussian filtered"
    }
    
    print(f"Comparing pulse shapes for {network.name}:")
    print("-" * 60)
    
    for shape_id, shape_name in pulse_shapes.items():
        params['pulse_shape'] = shape_id
        
        causality_mv, reciprocity_mv, passivity_mv = quality_check(**params)
        
        print(f"\nPulse Shape: {shape_name}")
        print(f"  Causality:   {causality_mv:.1f} mV")
        print(f"  Reciprocity: {reciprocity_mv:.1f} mV")
        print(f"  Passivity:   {passivity_mv:.1f} mV")


# Example 5: Full analysis with frequency and time domain metrics
def example_full_analysis():
    """Combine frequency and time domain metrics"""
    
    from ieee_p370_python import quality_check_frequency_domain
    
    # Load network
    network = rf.Network('example.s2p')
    freq = network.f
    sdata = np.transpose(network.s, (1, 2, 0))
    nf = len(freq)
    port_num = network.nports
    
    print(f"Complete IEEE P370 Analysis for {network.name}")
    print("=" * 60)
    
    # Frequency domain metrics
    causality_freq, reciprocity_freq, passivity_freq = quality_check_frequency_domain(
        sdata, nf, port_num
    )
    
    print("Frequency Domain Metrics:")
    print(f"  Causality:   {causality_freq:.2f}%")
    print(f"  Reciprocity: {reciprocity_freq:.2f}%")
    print(f"  Passivity:   {passivity_freq:.2f}%")
    print()
    
    # Time domain metrics
    causality_time, reciprocity_time, passivity_time = quality_check(
        freq=freq,
        sdata=sdata,
        port_num=port_num,
        data_rate=25.0,
        sample_per_ui=64,
        rise_per=0.35,
        pulse_shape=1,
        extrapolation_method=1
    )
    
    print("Time Domain Metrics (25 Gbps):")
    print(f"  Causality:   {causality_time:.1f} mV")
    print(f"  Reciprocity: {reciprocity_time:.1f} mV")
    print(f"  Passivity:   {passivity_time:.1f} mV")
    print()
    
    # Overall assessment
    print("Overall Assessment:")
    freq_score = (causality_freq + reciprocity_freq + passivity_freq) / 3
    
    if freq_score > 95 and causality_time < 5:
        print("  ✓ Excellent S-parameter quality")
    elif freq_score > 90 and causality_time < 10:
        print("  ✓ Good S-parameter quality")
    else:
        print("  ⚠ S-parameter quality needs improvement")


if __name__ == "__main__":
    print("IEEE P370 Time Domain Quality Check Examples")
    print("=" * 60)
    print()
    
    # Run examples (comment out those without required files)
    
    # example_from_file()
    example_synthetic()
    # example_batch_processing()
    # example_pulse_shapes()
    # example_full_analysis()
