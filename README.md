# sow-reap-paraphrasing

CODE UPDATED ON 5.16.2020. PLEASE USE UPDATED VERISON. 

Contains data/code for the paper "Neural Syntactic Preordering for Controlled Paraphrase Generation" (ACL 2020).
https://arxiv.org/abs/2005.02013

Additional data/resources/trained models are available at https://drive.google.com/drive/folders/1jfE9fUqn8NtLuugJlSPgnGqcvwt1nye2?usp=sharing
The link contains training data to train new models, and also trained models for both SOW and REAP.

Environment base is Python 3.6. Also see requirements.txt.
# Training new models
1. Download training data from the google drive. Keep data folder in the main folder. 
2. Download resources from the google drive. Keep in the main folder.
Training SOW model
Data files train_sow.hdf5 and dev_sow.hdf5 correspond to the training and dev files for SOW. These contain phrase level inputs and outputs, with exactly two constituents abstracted out. We train a seqq2seq transformer model to learn this transduction. Run the following commands from the project root folder to train the sow model:

```
export PYTHONPATH=./
python sow/train.py
```

Training REAP model
Data files train.hdf5 and dev.hdf5 correspond to the training and dev files for REAP. This model learns a seq2seq model to paraphrase the input sentence, additionally conditioned on an input reordering, that indicates the desired order of content in the output sentence. Run the following commands from the project root folder to train the reap model:

```
export PYTHONPATH=./
python reap/train.py
```

# Inference
1. Download resources from the google drive. Keep in the main folder.
2. To use trained models, download from the google drive. Change model location in the arguments to the trained model location.
Paraphrases can be generated using three schemes:
1. Baseline seq2seq that does not include any reorder information. Run generate_paraphrases_baseline.py. See sample_test_baseline.txt for sample input file. (Use the PTB tokenizer to tokenize the file before running the system, the sample file included with the code is tokenized already.)


```
java -mx4g -cp "*" edu.stanford.nlp.process.PTBTokenizer -preserveLines < sample_test_baseline.txt > sample_test_baseline.tok
python generate_paraphrases_baseline.py
```

2. REAP model with ground truth ordering. See sample_test_gt_reap.txt for sample input file required. The file contains sentence_1, sentence_2, sentence_1_reordering, sentence_2_reordering. See processing/get_ground_truth_alignments.py to generate this sample data (will be added soon).

```python generate_paraphrases_gt_reap.py```

3. Full SOW-REAP model. This first produces k reorderings for the input sentence using SOW, then generates a paraphrase correspondig to each of those reorderings (using REAP). 
See sample_test_sow_reap.txt for sample input file required. We use the stanford nlp parser to generate this. To generate this file for your custom dataset, run the following command (from the stanford parser folder) on a file with the same input scheme as the sample_test_baseline.tok. When using your own test data, tokeinize the file separately (command above) before running the parser, otherwise some of the future code breaks or produces non-sensical outputs. 

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,parse  -preserveLines -ssplit.eolonly true -outputFormat text -file sample_test_baseline.txt
```

The output from the stanford nlp parser serves as an input to our SOW-REAP generator. To generate paraphrases: 

```python generate_paraphrases_sow_reap.py```
