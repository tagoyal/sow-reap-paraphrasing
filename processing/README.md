This folder contains code to create training data for SOW and REAP models. This code uses the files in the resources folder at https://drive.google.com/drive/folders/1jfE9fUqn8NtLuugJlSPgnGqcvwt1nye2?usp=sharing 
To use this code, download the resources folder and place it in the root directory of this project. 

In order to generate data, follow the following three steps:

# Filtering data 
For the experiments in the paper, we used the ParaNMT-50M dataset. In that dataset, a large chunk of paraphrase pairs were quite similar to each other structurally. We therefore filter out these examples to only retain pairs with stuctural changes. 

To filter custom data, the code requires the following files:
1) An input file that contains the paraphrase pairs text. See sample_text_baseline.txt in the root repository for format.
2) Glove embedding file (download from https://nlp.stanford.edu/projects/glove/)
3) Output file path

Run ```python run_filter.py --input_file [input_file] --output_file [output_file] --wordEmbed_file [wordEmbed_file]```

The threshold used in our code is pretty stringent and can be adjusted to ensure that reasonably large data size is retained.

Next, we use the output file generated from the previous step to create training data to train both the SOW and the REAP models. The instructions for these are outlined below.

# Training data for SOW
The SOW model is used to transform abstracted phrases to guide generation. Training the SOW model requires aligning phrase pairs using ELMO. See paper for details. This code uses allennlp to get the ELMO vectors. We use the above generated output file for this.

Run ```sh create_sow_data.sh``` to generate the data. The following file paths need to be given/changed:

1) It requires the input file with the constituency parse tree for each sentence. The format is same at that of sample_test_sow_reap.txt file in the root directory. To get this, we use the Stanford CoreNLP parser. Look at the main README for instructions on how to generate this file.

2) The code also requires ELMO representations for each input sentence. For these, download the weights.hdf5 and options.hdf5 file for ELMO and place it in a folder. Set ELMO_WEIGHTS_FOLDER as the path to this folder.

3) The output folder where the final files should be saved. Files are saved in the hdf5 format and used by the sow/train.py code. 

Note that the code internally generates and uses an idf dictionary from the given input data. Therefore, it works best with large amounts of data. Else, you can generate a idf dictionary from other available data and replace in the code.

Along with the final output files, the code produces other intermediate generation files that can be studied (saved in the intermediate folder, path can be specified in sh create_sow_data.sh):
a) phrase_alignments.out for phrase level alignments between paraphrase pairs. 
b) Two files, one with reordering between the two abstracted subphrases (rules_with_reordering.out) and one without reordering between these abstracted phrases (rules_no_reordering.out). The final training data is constructed using both these.

# Training data for REAP
The Reap data is constructed by aligning words along the dependency tree of the paraphrase pairs using the bert representations. See paper for details. This code uses the BERTscore library to get the word level alignment. We slightly modify this library. To install, run the following command in the resources/bert_score folder ```pip install .```

Run ```sh create_reap_data.sh``` to generate the data. The following file paths need to be given/changed:

1)  It requires the input file with the dependency parse tree for each sentence. The format is same at that of sample_test_sow_reap.txt file in the root directory. THIS IS SAME AS THE INPUT FILE FOR SOW DATA CREATION USED ABOVE
2) The output folder where the final files should be saved. Files are saved in the hdf5 format and used by the reap/train.py code. 

