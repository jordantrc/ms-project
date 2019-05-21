# sample_visualization.py

import csv
import PIL.Image as Image
import numpy as np
import sys

layer_dimensions = [
                [64, 16],
                [128, 16],
                [256, 8],
                [512, 4],
                [512, 2]
]


def expand_bitmap(bitmap, dims, factor):
    '''expands the bitmap by factor'''
    new_bitmap = []
    for y in range(dims[0] * factor):
        new_bitmap.append([0] * dims[1] * factor)

    for y in range(dims[0]):
        for x in range(dims[1]):
            pixel = bitmap[y][x]
            for y_n in range(factor * y, factor * (y + 1)):
                for x_n in range(factor * x, factor * (x + 1)):
                    new_bitmap[y_n][x_n] = pixel

    return new_bitmap


file_name = sys.argv[1]
layer = int(sys.argv[2])

with open(file_name, newline='') as fd:
    reader = csv.reader(fd, dialect='excel')
    header = next(reader)
    first_sample = next(reader)

dims = layer_dimensions[layer]
bitmap = []
for y in range(dims[0]):
    bitmap.append([0] * dims[1])

first_sample = first_sample[2:]
print("first sample = %s" % first_sample)

for i, v in enumerate(first_sample):
    row_num = int(i / dims[1])
    col_num = int(i % dims[1])
    print("v = %s (%s, %s)" % (v, row_num, col_num))
    bitmap[row_num][col_num] = int(float(v) * 255)

# expand the bitmap
bitmap = expand_bitmap(bitmap, dims, 10)

img = Image.new('L', (dims[1] * 10, dims[0] * 10), color=0)
pixels = img.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixels[i,j] = bitmap[j][i]

output_file = file_name + ".bmp"
img.save(output_file, "BMP")
