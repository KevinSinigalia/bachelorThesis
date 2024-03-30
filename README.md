on error due to mpi4py: https://github.com/theislab/cellrank/issues/864

MOST IMPORTANT SCRIPTS AND NOTEBOOKS:
1) gridSearch.ipynb
2) PCMCI_getF1Scores.py
3) PCMCI_loop_numberComponents_pcAlpha.py
since other scripts are based on them



scripts and notebooks belonging together:

subnetworkLinksPCMCI.py <---> subnetworkLinks.ipynb

PCMCI+_f1Scorey.py <---> gridSearch_PCMCIplus_example.ipynb

getF1scores_selectComponents.py <---> subnetworkComponents.ipynb (or getOptimalCompsComb.ipynb)

PCMCI_getF1Scores.py, PCMCI_further_metrics.py <---> gridSearch.ipynb

PCMCI_loop_numberComponents_pcAlpha.py just creates results


----------------------------------------------------


PCMCI_loop_numberComponents_pcAlpha.py runs PCMCI for different pc_alpha and number components values, which can be passed as a list

PCMCI_F1-scores_subnetwork.py calculates the subnetworks and then the F1-scores based on the subnetworks

PCMCI_selectComponents.py + PCMCI_F1-scores_subnetwork.py runs PCMCI on specified components and then calculates the F1-scores


PCMCI+_f1Scores example how to get F1-scores from PCMCI+ results, since PCMCI has no mci_alpha and different result dictionary keys



gridSearch.ipynb visualization of the F1 or further metric scores for all used combinations of pc_alpha, mci_alpha, number components (replace F1-scores path with other path containing the distance measurement scores)

subnetworkLinks.ipynb visualization of subnetwork and achieved metric scores

subnetworkLinksPCMCIplus_example.ipynb and gridSearch_PCMCIplus_example.ipynb are examples how to do this for PCMCI+ results

