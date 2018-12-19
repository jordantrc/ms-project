import numpy as np 
import math#,cv2, imageio
import time

def heapify(arr, n, i): 
    largest = i # Initialize largest as root 
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
  
    # See if left child of root exists and is 
    # greater than root 
    if l < n and arr[i][0] < arr[l][0]: 
        largest = l 
  
    # See if right child of root exists and is 
    # greater than root 
    if r < n and arr[largest][0] < arr[r][0]: 
        largest = r 
  
    # Change root, if needed 
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i] # swap 
  
        # Heapify the root. 
        heapify(arr, n, largest)

def adjacency_peak(matrix, x, y):
	cur_val = matrix[x, y]
	w = matrix.shape[0]
	h = matrix.shape[1]
	if(x-1 >= 0 and matrix[x-1, y] >= cur_val):
		return False
	if(x+1 < w and matrix[x+1, y] >= cur_val):
		return False
	if(y-1 >= 0 and matrix[x, y-1] >= cur_val):
		return False
	if(y+1 < h and matrix[x, y+1] >= cur_val):
		return False
	
	if(x-1 >= 0 and y-1 >= 0 and matrix[x-1, y-1] >= cur_val):
		return False
	if(x+1 < w and y-1 >= 0 and matrix[x+1, y-1] >= cur_val):
		return False
	if(x-1 >=0 and y+1 < h and matrix[x-1, y+1] >= cur_val):
		return False
	if(x+1 < w and y+1 < h and matrix[x+1, y+1] >= cur_val):
		return False

	return True

def heap_collapse(alist, matrix, threshold=7.5, n=5):
	'''
	Perfrom a heap sort. Pass through the list to find the top N points that aren't within a distance
	of previous points. Should perform a thresholding approach to set values below a threshold to 0.
	'''
	alist_size = len(alist)

	threshold = matrix.shape[1]/7.5
	
	for i in range(alist_size, -1, -1): 
		heapify(alist, alist_size, i) 

	selected_peaks = []
	
	while(len(selected_peaks) < n and alist_size > 1):
		max_val = alist[0]
		alist[0], alist[alist_size-1] = alist[alist_size-1], alist[0]
		alist_size -= 1
		heapify(alist, alist_size, 0) 

		#print(max_val)

		#check peakness:
		cur_p = max_val[1]
		if(adjacency_peak(matrix, cur_p[0], cur_p[1])):
			not_in_range = True
			for i in range(len(selected_peaks)):
				#check distance
				other_p = selected_peaks[i][1]
				if( math.sqrt((cur_p[0] - other_p[0])**2 +(cur_p[1] - other_p[1])**2) < threshold):
					not_in_range = False
			if(not_in_range):
				selected_peaks.append(max_val)
	return selected_peaks

def peak_finding_fast(matrix_3d, n=10):
	num_frames = matrix_3d.shape[0]
	w = matrix_3d.shape[1]
	h = matrix_3d.shape[2]

	step = w/3

	avg = np.average(matrix_3d)
	peak_ordering = np.zeros(num_frames*n).reshape(n, num_frames)

	for t in range(num_frames):
		heap_list = []

		for x in range(0, w, step):
			if(x+step > w):
				x = w -step
			for y in range(0, h, step):
				if(y+step > h):
					y = h - step

				region = matrix_3d[t, x:x+step, y:y+step]

				pos = np.unravel_index(np.argmax(region), dims=(step, step))
				value = region[pos[0], pos[1]]
				location = (pos[0]+x, pos[1]+y)
				if(value > avg):
					heap_list.append((value, location))

		if(len(heap_list) > 0):

			select_peaks = heap_collapse(heap_list,matrix_3d[t], n=n)
			for p in range(len(select_peaks)):
				peak_ordering[p, t] = select_peaks[p][0]

	return peak_ordering


if __name__ == '__main__':
	activations = np.load("activations_1.npy")
	print(activations.shape)

	t_s = time.time()
	peaks2 = peak_finding_fast(activations[0,:, :, :, 0])
	print("peak_finding:", time.time()-t_s)
	print(peaks2.shape)

	print(peaks2[:, 50])

	t_s = time.time()
	activations_reshape = activations.reshape((activations.shape[0], -1))
	activation_divisions = [np.max(activations_reshape, axis=1)]
	print("max_find:", time.time()-t_s)

	t_s = time.time()

	step = activations.shape[1]/2
		
	for n in range(0, activations.shape[1], step):
		for m in range(0, activations.shape[2], step):
			restruct = activations[:, n:n+step, m:m+step].reshape((activations.shape[0], -1))
			activation_divisions.append(np.max(restruct, axis=1))
	print("quad_find:", time.time()-t_s)
