INPUT_FILE=../sample_test_sow_reap.txt
ELMO_WEIGHTS_FOLDER=../../../../Temporal/bilm-tf-master/tests/fixtures/model/
INTERMEDIATE_FOLDER=sow_intermediate
OUTPUT_FOLDER=../data/custom

mkdir $INTERMEDIATE_FOLDER
python get_phrase_list.py --input_file $INPUT_FILE --output_folder $INTERMEDIATE_FOLDER
python get_elmo_embeds.py --elmo_data_dir $ELMO_WEIGHTS_FOLDER --input_file $INPUT_FILE --output_folder $INTERMEDIATE_FOLDER
python get_phrase_alignment.py --output_folder $INTERMEDIATE_FOLDER
python create_rules2.py --input_folder $INTERMEDIATE_FOLDER
python convert_hdf5_sow.py --input_folder $INTERMEDIATE_FOLDER --out_path $OUTPUT_FOLDER