# evaluate_predict_ret.py
#
# Provides accuracy summary from a predict_ret.txt
# file produced by predict_c3d_ucf101.py
#
# Usage: evaluate_predict_ret.py <filename>

import sys

def main():
    '''main function'''
    if len(sys.argv) != 2:
        print_help()
        sys.exit(1)
    file_name = sys.argv[1]

    # open and read the file
    with open(file_name) as fd:
        raw_text = fd.read()
        lines = raw_text.splitlines()

    # parse the contents of the file
    total_predictions = 0
    correct_clip_1 = 0
    correct_clip_5 = 0
    video_predictions = {}

    for l in lines:
        total_predictions += 1
        data = l.split(",")
        clip = data[0]
        video = '_'.join(clip.split('_')[0:3])
        actual = data[1].strip()
        prediction = data[3].strip()
        top_5 = data[5].replace('[', '').replace(']', '').strip().split(' ')
        while '' in top_5:
            top_5.remove('')

        # video predictions
        if video not in video_predictions.keys():
            video_predictions[video] = [0, 0]

        # add to clip accumulators
        video_predictions[video][0] += 1
        if actual == prediction:
            correct_clip_1 += 1
            correct_clip_5 += 1
            video_predictions[video][1] += 1
        else:
            if actual in top_5:
                correct_clip_5 += 1
    
    # calculate video accuracy
    video_total = 0
    video_correct = 0
    for k in video_predictions:
        video_total += video_predictions[k][0]
        video_correct += video_predictions[k][1]

    # print results
    print("Accuracy Summary:")
    print("Clip accuracy: %s" % (correct_clip_1 / total_predictions))
    print("Clip@5 accuracy: %s" % (correct_clip_5 / total_predictions))
    print("Video accuracy: %s" % (video_correct / video_total))       

def print_help():
    '''prints help'''
    print("Usage: evaluate_predict_ret.py <filename>")


if __name__ == "__main__":
    main()
