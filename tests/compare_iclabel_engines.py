import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr

# ### Using ICLabel with Different Engines

# ```python
# from eegprep import pop_loadset, iclabel

# # Load an EEG file
# EEG = pop_loadset('./data/eeglab_data_with_ica_tmp.set')

# # Apply ICLabel with the default Python implementation
# EEG_python = iclabel(EEG, algorithm='default', engine=None)
# EEG_matlab = iclabel(EEG, algorithm='default', engine='matlab')
# ```

# ### Running the Comparison Script

# The `test_iclabel_engines.py` script can be used to compare the results of applying ICLabel with different engines:

# ```bash
# python test_iclabel_engines.py your_eeg_file.set --output_dir results --algorithm default
# ```


# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import pop_loadset, pop_saveset, iclabel

def compare_iclabel_engines(input_file, output_dir=None, engines=None, algorithm='default'):
    """
    Compare the results of applying ICLabel with different engines.
    
    Parameters:
    -----------
    input_file : str
        Path to the input EEG file
    output_dir : str or None
        Directory to save output files. If None, files are not saved.
    engines : list or None
        List of engines to compare. If None, all available engines are used.
        Options are: None (Python), 'matlab', 'octave'
    algorithm : str
        Algorithm to use for classification, passed to the MATLAB/Octave implementation.
    
    Returns:
    --------
    results : dict
        Dictionary containing the results of each engine
    """
    if engines is None:
        engines = [
            None,  # Python implementation
            'matlab',
        ]
    
    # Load the input EEG file
    print(f"Loading EEG file: {input_file}")
    EEG = pop_loadset(input_file)
    
    # Apply ICLabel with different engines
    results = {}
    for engine in engines:
        try:
            engine_name = engine if engine is not None else 'python'
            print(f"Applying ICLabel with engine={engine_name}, algorithm={algorithm}")
            EEG_result = iclabel(EEG.copy(), algorithm=algorithm, engine=engine)
            
            # Store the result
            results[engine_name] = EEG_result
            
            # Save the result if output_dir is specified
            if output_dir is not None:
                output_file = os.path.join(output_dir, f"iclabel_{engine_name}_{algorithm}.set")
                pop_saveset(EEG_result, output_file)
                print(f"Saved result to: {output_file}")
        except Exception as e:
            print(f"Error applying ICLabel with engine={engine_name}, algorithm={algorithm}: {e}")
    
    return results

def compare_classifications(results):
    """
    Compare the classifications from different engines.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the results of each engine
    
    Returns:
    --------
    comparisons : dict
        Dictionary containing the comparison metrics
    """
    engines = list(results.keys())
    n_engines = len(engines)
    
    comparisons = {}
    
    # Compare each pair of engines
    for i in range(n_engines):
        for j in range(i+1, n_engines):
            engine1 = engines[i]
            engine2 = engines[j]
            
            # Get the classifications
            classifications1 = results[engine1]['etc']['ic_classification']['ICLabel']['classifications']
            classifications2 = results[engine2]['etc']['ic_classification']['ICLabel']['classifications']
            
            # Calculate correlation
            correlations = []
            for comp_idx in range(classifications1.shape[0]):
                corr, _ = pearsonr(classifications1[comp_idx], classifications2[comp_idx])
                correlations.append(corr)
            
            # Calculate mean absolute difference
            mean_abs_diff = np.mean(np.abs(classifications1 - classifications2))
            
            # Store the comparison metrics
            comparisons[(engine1, engine2)] = {
                'correlations': correlations,
                'mean_correlation': np.mean(correlations),
                'mean_abs_diff': mean_abs_diff
            }
    
    return comparisons

def plot_comparisons(results, comparisons, output_dir=None, algorithm='default'):
    """
    Plot the comparisons between different engines.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the results of each engine
    comparisons : dict
        Dictionary containing the comparison metrics
    output_dir : str or None
        Directory to save output files. If None, files are not saved.
    algorithm : str
        Algorithm used for classification
    """
    engines = list(results.keys())
    n_engines = len(engines)
    
    # Plot the classifications for each engine
    plt.figure(figsize=(15, 5 * n_engines))
    
    for i, engine in enumerate(engines):
        plt.subplot(n_engines, 1, i+1)
        classifications = results[engine]['etc']['ic_classification']['ICLabel']['classifications']
        classes = results[engine]['etc']['ic_classification']['ICLabel']['classes']
        
        plt.imshow(classifications, aspect='auto', cmap='viridis')
        plt.colorbar(label='Probability')
        plt.yticks(range(len(classes)), classes)
        plt.title(f"ICLabel Classifications - Engine: {engine}, Algorithm: {algorithm}")
        plt.xlabel('Component')
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'iclabel_classifications_{algorithm}.png'))
    
    # Plot the correlations between engines
    plt.figure(figsize=(15, 5 * len(comparisons)))
    
    for i, ((engine1, engine2), metrics) in enumerate(comparisons.items()):
        plt.subplot(len(comparisons), 1, i+1)
        plt.bar(range(len(metrics['correlations'])), metrics['correlations'])
        plt.axhline(metrics['mean_correlation'], color='r', linestyle='--', label=f"Mean: {metrics['mean_correlation']:.3f}")
        plt.title(f"Correlation between {engine1} and {engine2}")
        plt.xlabel('Component')
        plt.ylabel('Correlation')
        plt.legend()
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'iclabel_correlations_{algorithm}.png'))

def main():
    """
    Main function to run the comparison.
    """
    parser = argparse.ArgumentParser(description='Compare ICLabel engines')
    parser.add_argument('input_file', help='Path to the input EEG file')
    parser.add_argument('--output_dir', help='Directory to save output files')
    parser.add_argument('--algorithm', default='default', help='Algorithm to use for classification')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare ICLabel engines
    results = compare_iclabel_engines(args.input_file, args.output_dir, algorithm=args.algorithm)
    
    # Compare classifications
    if len(results) > 1:
        comparisons = compare_classifications(results)
        
        # Plot comparisons
        plot_comparisons(results, comparisons, args.output_dir, args.algorithm)
        
        # Wait for user input before closing the plot
        plt.show()
        
        # Print summary
        print("\nSummary of comparisons:")
        for (engine1, engine2), metrics in comparisons.items():
            print(f"Comparison between {engine1} and {engine2}:")
            print(f"  Mean correlation: {metrics['mean_correlation']:.3f}")
            print(f"  Mean absolute difference: {metrics['mean_abs_diff']:.3f}")
    else:
        print("Not enough results to compare.")

if __name__ == '__main__':
    main() 