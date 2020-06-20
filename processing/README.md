This folder contains code to create training data for SOW and REAP models.

# Training data for SOW
Training the SOW model required aligning phrase pairs using ELMO. This code uses allennlp to get the ELMO vectors.

Run ```sh create_sow_data.sh``` to generate the data. The following file paths need to be given/changed:

1) It requires the input file (with the same scheme as the sample_test_sow_reap.txt file. Look at the main README for instructions on how to generate this file.)

2) The code also requires ELMO representations for each input sentence. For these, download the weights.hdf5 and options.hdf5 file for ELMO and place it in a folder. Set ELMO_WEIGHTS_FOLDER as the path to this folder.

This code produces two output files, one with reordering between the two bstracted subphrases (rules_with_reordering.out) and one without reordering between these abstracted phrases (rules_no_reordering.out). The final training data is constructed using both these.
The code internally generates and uses an idf dictionary from the given input data. Therefore, it works best with large amounts of data. Else, you can generate a idf dictionary from other available data and replace in the code.

Finally, the code also generated other intermediate files, phrase_alignments.out that can be studied to look at the alignemnts produced.
