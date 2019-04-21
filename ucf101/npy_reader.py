import numpy as np 

class BatchReaderNumpy():
	def __init__(self, dataset_file, batch_size, shuffle):
		file = np.load(dataset_file)
		self.data = file["data"]
		self.labels = file["label"]
		self.batch_size = batch_size
		self.current_position = 0
		self.shuffle = shuffle

	def size(self):
		return len(self.labels)

	def get_batch(self):
		if(self.shuffle):
			indexes = np.random.randint(0, len(self.data), self.batch_size)
			return self.data[indexes], self.labels[indexes]
		else:
			start = self.current_position
			end = (self.current_position+self.batch_size) % len(self.data)

			if(end < start):
				indexes = range(start, len(self.data)) + range(end)
			else:
				indexes = range(start, end)
			self.current_position = end

			return self.data[indexes], self.labels[indexes]

if __name__ == '__main__':
	r = BatchReaderNumpy("npy_files/ucf_50_train_0.npz", 10, False)
	
	for i in range(50):
		data, label = r.get_batch()
		print(data.shape)
		print(label)
