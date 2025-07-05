"""
IEEE P370 Integration Module for OpenSNPQual
This module provides integration with IEEE P370 MATLAB code
"""

import subprocess
import tempfile
import json
import os
from typing import Dict, Optional
import matlab.engine  # Requires MATLAB Python API


class IEEEP370Integration:
    """Integrate IEEE P370 MATLAB implementation"""
    
    def __init__(self, matlab_path: str = None):
        """
        Initialize P370 integration
        
        Args:
            matlab_path: Path to IEEE P370 MATLAB code directory
        """
        self.matlab_path = matlab_path or os.environ.get('IEEE_P370_PATH', '')
        self.eng = None
        
    def start_matlab_engine(self):
        """Start MATLAB engine for P370 calculations"""
        try:
            self.eng = matlab.engine.start_matlab()
            if self.matlab_path:
                self.eng.addpath(self.matlab_path)
            return True
        except Exception as e:
            print(f"Failed to start MATLAB engine: {e}")
            return False
    
    def check_passivity_p370(self, s_params, freq):
        """
        Check passivity using IEEE P370 MATLAB implementation
        
        Args:
            s_params: S-parameter matrix (numpy array)
            freq: Frequency points (numpy array)
        
        Returns:
            Dict with passivity metrics
        """
        if not self.eng:
            return None
            
        try:
            # Convert to MATLAB format
            s_matlab = matlab.double(s_params.tolist())
            f_matlab = matlab.double(freq.tolist())
            
            # Call P370 passivity check
            # Assuming P370 function: [metric, report] = check_passivity(S, freq)
            metric, report = self.eng.check_passivity(s_matlab, f_matlab, nargout=2)
            
            return {
                'metric': float(metric),
                'report': str(report)
            }
        except Exception as e:
            print(f"P370 passivity check failed: {e}")
            return None
    
    def check_reciprocity_p370(self, s_params, freq):
        """Check reciprocity using IEEE P370"""
        if not self.eng:
            return None
            
        try:
            s_matlab = matlab.double(s_params.tolist())
            f_matlab = matlab.double(freq.tolist())
            
            # Call P370 reciprocity check
            metric, report = self.eng.check_reciprocity(s_matlab, f_matlab, nargout=2)
            
            return {
                'metric': float(metric),
                'report': str(report)
            }
        except Exception as e:
            print(f"P370 reciprocity check failed: {e}")
            return None
    
    def check_causality_p370(self, s_params, freq):
        """Check causality using IEEE P370"""
        if not self.eng:
            return None
            
        try:
            s_matlab = matlab.double(s_params.tolist())
            f_matlab = matlab.double(freq.tolist())
            
            # Call P370 causality check
            metric, report = self.eng.check_causality(s_matlab, f_matlab, nargout=2)
            
            return {
                'metric': float(metric),
                'report': str(report)
            }
        except Exception as e:
            print(f"P370 causality check failed: {e}")
            return None
    
    def close(self):
        """Close MATLAB engine"""
        if self.eng:
            self.eng.quit()


class ValidationTools:
    """Validate OpenSNPQual results against other tools"""
    
    @staticmethod
    def compare_with_simbeor(snp_file: str, simbeor_exe: str = None) -> Dict:
        """
        Compare results with Simbeor
        
        Args:
            snp_file: Path to S-parameter file
            simbeor_exe: Path to Simbeor executable
        
        Returns:
            Dict with comparison results
        """
        if not simbeor_exe or not os.path.exists(simbeor_exe):
            return {'error': 'Simbeor executable not found'}
        
        try:
            # Run Simbeor quality check
            cmd = [simbeor_exe, '-quality', snp_file, '-output', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse Simbeor output
                simbeor_data = json.loads(result.stdout)
                return {
                    'passivity': simbeor_data.get('passivity', {}),
                    'reciprocity': simbeor_data.get('reciprocity', {}),
                    'causality': simbeor_data.get('causality', {})
                }
            else:
                return {'error': f'Simbeor failed: {result.stderr}'}
        except Exception as e:
            return {'error': f'Simbeor comparison failed: {e}'}
    
    @staticmethod
    def compare_with_plts(snp_file: str, plts_path: str = None) -> Dict:
        """
        Compare results with Keysight PLTS
        
        Args:
            snp_file: Path to S-parameter file
            plts_path: Path to PLTS installation
        
        Returns:
            Dict with comparison results
        """
        # PLTS typically uses COM automation on Windows
        # This is a placeholder for PLTS integration
        try:
            import win32com.client
            
            # Initialize PLTS COM object
            plts = win32com.client.Dispatch("PLTS.Application")
            plts.LoadFile(snp_file)
            
            # Get quality metrics
            quality = plts.GetQualityMetrics()
            
            return {
                'passivity': quality.Passivity,
                'reciprocity': quality.Reciprocity,
                'causality': quality.Causality
            }
        except Exception as e:
            return {'error': f'PLTS comparison failed: {e}'}
    
    @staticmethod
    def generate_comparison_report(snp_file: str, 
                                  opensnpqual_results: Dict,
                                  p370_results: Dict = None,
                                  simbeor_results: Dict = None,
                                  plts_results: Dict = None) -> str:
        """
        Generate comparison report between different tools
        
        Returns:
            Markdown formatted comparison report
        """
        report = f"# Quality Metrics Comparison Report\n\n"
        report += f"**File:** {snp_file}\n\n"
        
        # Create comparison table
        report += "| Metric | OpenSNPQual | IEEE P370 | Simbeor | PLTS |\n"
        report += "|--------|-------------|-----------|---------|------|\n"
        
        metrics = ['passivity', 'reciprocity', 'causality']
        
        for metric in metrics:
            row = [metric.capitalize()]
            
            # OpenSNPQual result
            osnp_val = opensnpqual_results.get(f'{metric}_freq', 'N/A')
            row.append(f"{osnp_val:.4f}" if isinstance(osnp_val, (int, float)) else str(osnp_val))
            
            # P370 result
            if p370_results and metric in p370_results:
                p370_val = p370_results[metric].get('metric', 'N/A')
                row.append(f"{p370_val:.4f}" if isinstance(p370_val, (int, float)) else str(p370_val))
            else:
                row.append('N/A')
            
            # Simbeor result
            if simbeor_results and not simbeor_results.get('error'):
                simb_val = simbeor_results.get(metric, {}).get('value', 'N/A')
                row.append(f"{simb_val:.4f}" if isinstance(simb_val, (int, float)) else str(simb_val))
            else:
                row.append('N/A')
            
            # PLTS result
            if plts_results and not plts_results.get('error'):
                plts_val = plts_results.get(metric, 'N/A')
                row.append(f"{plts_val:.4f}" if isinstance(plts_val, (int, float)) else str(plts_val))
            else:
                row.append('N/A')
            
            report += f"| {' | '.join(row)} |\n"
        
        # Add notes
        report += "\n## Notes:\n"
        report += "- OpenSNPQual uses scikit-rf based implementation\n"
        report += "- IEEE P370 results from official MATLAB implementation\n"
        report += "- Simbeor and PLTS results require respective software installations\n"
        
        return report


# Example usage for validation
if __name__ == "__main__":
    # Example: Validate against IEEE P370
    p370 = IEEEP370Integration('/path/to/ieee-p370-matlab')
    if p370.start_matlab_engine():
        # Load S-parameters
        import skrf as rf
        network = rf.Network('example.s2p')
        
        # Get P370 results
        passivity = p370.check_passivity_p370(network.s, network.f)
        print(f"P370 Passivity: {passivity}")
        
        p370.close()
    
    # Example: Generate comparison report
    validation = ValidationTools()
    
    # Mock results for demonstration
    opensnpqual_results = {
        'passivity_freq': 0.012,
        'reciprocity_freq': 0.008,
        'causality_freq': 0.025
    }
    
    report = validation.generate_comparison_report(
        'example.s2p',
        opensnpqual_results
    )
    
    print(report)
