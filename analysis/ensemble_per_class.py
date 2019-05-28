# ensemble_per_class.py

import csv
import sys

input_file = sys.argv[1]
classes_file = sys.argv[2]


def class_category(cl, cl_list):
    for r in cl_list:
        if cl == r[0]:
            return r[2]

def class_name(cl, cl_list):
    for r in cl_list:
        if cl == r[0]:
            return r[1]

predict_data = []
with open(input_file, newline='') as fd:
    csv_r = csv.reader(fd, dialect='excel')
    header = next(csv_r)
    for row in csv_r:
        predict_data.append(row)

class_data = []
with open(classes_file, newline='') as fd:
    csv_r = csv.reader(fd, dialect='excel')
    header = next(csv_r)
    for row in csv_r:
        class_data.append(row)


# summarize results
# data structure format is:
# [class_label: [total, model0_num_correct, model1_num_correct, ..., ensemble_num_correct]]
# [category: [total, model0_num_correct, model1_num_correct, ..., ensemble_num_correct]]
summary_class = {}
summary_category = {}

for r in predict_data:
    true_class = str(r[0])
    category = class_category(true_class, class_data)
    model = r[1]
    
    # set the index
    if model == 'ensemble':
        index = 7
    else:
        index = int(model) + 1

    place = r[2]
    pred_class = r[3]

    # only look at top predictions
    if place == '0':
        if true_class not in summary_class.keys():
            summary_class[true_class] = [1, 0, 0, 0, 0, 0, 0, 0]

        # only add to total once
        if model == '0':
            summary_class[true_class][0] += 1

        if pred_class == true_class:
            summary_class[true_class][index] += 1

        if category not in summary_category.keys():
            summary_category[category] = [1, 0, 0, 0, 0, 0, 0, 0]
        
        # only add to total once
        if model == '0':
            summary_category[category][0] += 1

        if pred_class == true_class:
            summary_category[category][index] += 1

# print summary
print("per-class accuracy:")
print("class,model,accuracy")
for k in sorted(summary_class.keys()):
    #print("%s = %s" % (k, summary_class[k]))
    class_total = float(summary_class[k][0])
    for i, v in enumerate(summary_class[k][1:]):
        if i == 6:
            model = "ensemble"
        else:
            model = i
        print("%s,%s,%05f" % (class_name(k, class_data), model, float(v) / class_total))

print("\n\nper-category accuracy:")
print("category,model,accuracy")
for k in sorted(summary_category.keys()):
    #print("%s = %s" % (k, summary_category[k]))
    category_total = float(summary_category[k][0])
    for i, v in enumerate(summary_category[k][1:]):
        if i == 6:
            model = "ensemble"
        else:
            model = i
        print("%s,%s,%05f" % (k, model, float(v) / category_total))
