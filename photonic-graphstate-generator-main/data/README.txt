The two most important files are the combined_classes.txt and linear-rank-width_LC-classes.txt

combined_classes.txt is combined data from Cabello et al. 'Optimal preparation of graph states' with <=10 qubits


linear-rank-width_LC-classes.txt classifies the number of emitters needed for a LC-class.
structure:
LC-class number; emitters; one emission order for this number of emitters (with respect to the labelling in combined_classes)

The zipped directory in the main folder. There is also a file all_lrw.txt that classifies all permutations of graphs with <= 8 qubits.
with respect to the graphs from combined_classes and permutations from the iteration package 
one could check all orderings that needs a specific amount of emitters and check for their em-CNOT cost.