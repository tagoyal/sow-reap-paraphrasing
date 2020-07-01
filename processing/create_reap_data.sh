INPUT_FILE=../sample_test_sow_reap.txt
INTERMEDIATE_FOLDER=reap_intermediate
OUTPUT_FOLDER=../data/custom

mkdir $INTERMEDIATE_FOLDER
python create_reap_data.py --input_file $INPUT_FILE --output_folder $INTERMEDIATE_FOLDER
python convert_hdf5_reap.py --input_folder $INTERMEDIATE_FOLDER --out_path $OUTPUT_FOLDER