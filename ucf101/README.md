Un-tar the JSON IAD files using the following command

	tar -zxf <iad file>.tar.gz

create list file indicating the location of the IADS to be read using the batch_json_reader.py file. You will need to do this for each train and test folder separateley:

	convert_files_to_batchable_format(<caffe.txt file>, <directory of JSON IADS>, <output_name>)

	convert_files_to_batchable_format('c3d_ucf101_train_split1.txt', 'generated_iads_ucf_75/generated_iads_train_75', 'ucf75train.list')

Include the reader in your code as follows. Will shuffle the data if training dataset

	reader = BatchJsonRead(<filename>, <batch_size>, <network_depth>, <train>)

	reader = BatchJsonRead('ucf75train.list', 30, 0, True)
	data, label = reader.get_batch()
