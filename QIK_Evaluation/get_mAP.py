import numpy as np
import pickle
import argparse
from pycocotools.coco import COCO
from prettytable import PrettyTable

# Global variables
coco = None
image_list = []
captions_lst = []
image_subset = []
ground_truth_dict = None
pre_computed_results = None
category_combination = None

# Local constants
SIMILARITY_THRESHOLD = .70
IMAGE_SET_PATH = "data/15K_Dataset.pkl"
PRE_COMPUTED_RESULTS_PATH = "pre_constructed_data/15K_Results.pkl"
OUTPUT_FILE = "data/QIK_Output_Combined.txt"
DATA_DIR = 'data'
DATA_TYPE = '2017'
ANN_FILE = '{}/instances_{}.json'.format(DATA_DIR,DATA_TYPE)
CAPTIONS_FILE = '{}/captions_{}.json'.format(DATA_DIR,DATA_TYPE)
PRE_COMPUTED_GROUND_TRUTH_PATH = "pre_constructed_data/15K_Results.pkl"

def init():
    global coco, image_subset, pre_computed_results, ground_truth_dict

    # Loading annotations and creating an Index
    coco = COCO(ANN_FILE)

    # Loading the subset of images.
    image_subset = pickle.load(open(IMAGE_SET_PATH, "rb"))

    # Loading the precomputed ground truth.
    ground_truth_dict = pickle.load(open(PRE_COMPUTED_GROUND_TRUTH_PATH, "rb"))

    # Loading the precomputed results.
    pre_computed_results = pickle.load(open(PRE_COMPUTED_RESULTS_PATH, "rb"))

def evaluate(query_lst):

    # length of query images
    query_lst_len = len(query_lst)

    # 1) QIK Results List.
    qik_time_lst = []
    qik_2_relevance_lst = []
    qik_4_relevance_lst = []
    qik_8_relevance_lst = []
    qik_16_relevance_lst = []

    # 2) DIR Results List.
    dir_time_lst = []
    dir_2_relevance_lst = []
    dir_4_relevance_lst = []
    dir_8_relevance_lst = []
    dir_16_relevance_lst = []

    # 3) LIRE Results List.
    lire_time_lst = []
    lire_2_relevance_lst = []
    lire_4_relevance_lst = []
    lire_8_relevance_lst = []
    lire_16_relevance_lst = []

    # 4) DELF Results List.
    delf_time_lst = []
    delf_2_relevance_lst = []
    delf_4_relevance_lst = []
    delf_8_relevance_lst = []
    delf_16_relevance_lst = []

    # 5) Deep Vision Results List.
    dv_time_lst = []
    dv_2_relevance_lst = []
    dv_4_relevance_lst = []
    dv_8_relevance_lst = []
    dv_16_relevance_lst = []

    # 6) CroW Results List.
    crow_time_lst = []
    crow_2_relevance_lst = []
    crow_4_relevance_lst = []
    crow_8_relevance_lst = []
    crow_16_relevance_lst = []

    # 7) CSQ Results List.
    csq_time_lst = []
    csq_2_relevance_lst = []
    csq_4_relevance_lst = []
    csq_8_relevance_lst = []
    csq_16_relevance_lst = []

    # 8) QIK Show and Tell Results List.
    qik_show_and_tell_time_lst = []
    qik_show_and_tell_2_relevance_lst = []
    qik_show_and_tell_4_relevance_lst = []
    qik_show_and_tell_8_relevance_lst = []
    qik_show_and_tell_16_relevance_lst = []

    for query_image in query_lst:

        if query_image not in pre_computed_results:
            query_lst_len -= 1
            continue

        # Defining the ground truth.
        if query_image not in ground_truth_dict:
            query_lst_len -= 1
            continue

        ground_truth = ground_truth_dict[query_image]

        # Get QIK results
        qik_results = pre_computed_results[query_image]["qik_results"]
        qik_time_lst.append(pre_computed_results[query_image]["qik_time"])

        # Get QIK Show and Tell results
        qik_show_and_tell_results = pre_computed_results[query_image]["qik_show_and_tell_results"]
        qik_show_and_tell_time_lst.append(pre_computed_results[query_image]["qik_show_and_tell_time"])

        # Get DIR results
        dir_results = pre_computed_results[query_image]["dir_results"]
        dir_time_lst.append(pre_computed_results[query_image]["dir_time"])

        # Get LIRE results
        lire_results = pre_computed_results[query_image]["lire_results"]
        lire_time_lst.append(pre_computed_results[query_image]["lire_time"])

        # Get DELF results
        delf_results = pre_computed_results[query_image]["delf_results"]
        delf_time_lst.append(pre_computed_results[query_image]["delf_time"])

        # Deep Vision results
        dv_results = pre_computed_results[query_image]["dv_results"]
        dv_time_lst.append(pre_computed_results[query_image]["dv_time"])

        # CROW results
        crow_results = pre_computed_results[query_image]["crow_results"]
        crow_time_lst.append(pre_computed_results[query_image]["crow_time"])

        # CSQ results
        csq_results = pre_computed_results[query_image]["csq_results"]
        csq_time_lst.append(pre_computed_results[query_image]["csq_time"])

        # Computing the mAP values.
        if len(qik_results) <= 0 or len(qik_show_and_tell_results) <=0:
            # Decrementing the count from the list of images.
            query_lst_len -= 1
            continue

        # k = 2
        qik_2_relevance_lst.append(get_binary_relevance(qik_results[:2], ground_truth))
        qik_show_and_tell_2_relevance_lst.append(get_binary_relevance(qik_show_and_tell_results[:2], ground_truth))
        dir_2_relevance_lst.append(get_binary_relevance(dir_results[:2], ground_truth))
        lire_2_relevance_lst.append(get_binary_relevance(lire_results[:2], ground_truth))
        delf_2_relevance_lst.append(get_binary_relevance(delf_results[:2], ground_truth))
        dv_2_relevance_lst.append(get_binary_relevance(dv_results[:2], ground_truth))
        crow_2_relevance_lst.append(get_binary_relevance(crow_results[:2], ground_truth))
        csq_2_relevance_lst.append(get_binary_relevance(csq_results[:2], ground_truth))

        # k=4
        qik_4_relevance_lst.append(get_binary_relevance(qik_results[:4], ground_truth))
        qik_show_and_tell_4_relevance_lst.append(get_binary_relevance(qik_show_and_tell_results[:4], ground_truth))
        dir_4_relevance_lst.append(get_binary_relevance(dir_results[:4], ground_truth))
        lire_4_relevance_lst.append(get_binary_relevance(lire_results[:4], ground_truth))
        delf_4_relevance_lst.append(get_binary_relevance(delf_results[:4], ground_truth))
        dv_4_relevance_lst.append(get_binary_relevance(dv_results[:4], ground_truth))
        crow_4_relevance_lst.append(get_binary_relevance(crow_results[:4], ground_truth))
        csq_4_relevance_lst.append(get_binary_relevance(csq_results[:4], ground_truth))

        # k=8
        qik_8_relevance_lst.append(get_binary_relevance(qik_results[:8], ground_truth))
        qik_show_and_tell_8_relevance_lst.append(get_binary_relevance(qik_show_and_tell_results[:8], ground_truth))
        dir_8_relevance_lst.append(get_binary_relevance(dir_results[:8], ground_truth))
        lire_8_relevance_lst.append(get_binary_relevance(lire_results[:8], ground_truth))
        delf_8_relevance_lst.append(get_binary_relevance(delf_results[:8], ground_truth))
        dv_8_relevance_lst.append(get_binary_relevance(dv_results[:8], ground_truth))
        crow_8_relevance_lst.append(get_binary_relevance(crow_results[:8], ground_truth))
        csq_8_relevance_lst.append(get_binary_relevance(csq_results[:8], ground_truth))

        # k=16
        qik_16_relevance_lst.append(get_binary_relevance(qik_results[:16], ground_truth))
        qik_show_and_tell_16_relevance_lst.append(get_binary_relevance(qik_show_and_tell_results[:16], ground_truth))
        dir_16_relevance_lst.append(get_binary_relevance(dir_results[:16], ground_truth))
        lire_16_relevance_lst.append(get_binary_relevance(lire_results[:16], ground_truth))
        delf_16_relevance_lst.append(get_binary_relevance(delf_results[:16], ground_truth))
        dv_16_relevance_lst.append(get_binary_relevance(dv_results[:16], ground_truth))
        crow_16_relevance_lst.append(get_binary_relevance(crow_results[:16], ground_truth))
        csq_16_relevance_lst.append(get_binary_relevance(csq_results[:16], ground_truth))

    if query_lst_len <= 0:
        print ("evaluate.py :: evaluate ::  Skipping evaluation (since no precomputed images were present) for :: %s" % (category_combination))
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    else:
        # 1) QIK
        # k=2
        qik_2_map = get_mAP(qik_2_relevance_lst)

        # k=4
        qik_4_map = get_mAP(qik_4_relevance_lst)

        # k=8
        qik_8_map = get_mAP(qik_8_relevance_lst)

        # k=16
        qik_16_map = get_mAP(qik_16_relevance_lst)

        # Average time
        qik_time_avg = get_average(qik_time_lst)

        # 2) DIR
        # k=2
        dir_2_map = get_mAP(dir_2_relevance_lst)

        # k=4
        dir_4_map = get_mAP(dir_4_relevance_lst)

        # k=8
        dir_8_map = get_mAP(dir_8_relevance_lst)

        # k=16
        dir_16_map = get_mAP(dir_16_relevance_lst)

        # Average time
        dir_time_avg = get_average(dir_time_lst)

        # 3) LIRE
        # k=2
        lire_2_map = get_mAP(lire_2_relevance_lst)

        # k=4
        lire_4_map = get_mAP(lire_4_relevance_lst)

        # k=8
        lire_8_map = get_mAP(lire_8_relevance_lst)

        # k=16
        lire_16_map = get_mAP(lire_16_relevance_lst)

        # Average time
        lire_time_avg = get_average(lire_time_lst)

        # 4) DELF
        # k=2
        delf_2_map = get_mAP(delf_2_relevance_lst)

        # k=4
        delf_4_map = get_mAP(delf_4_relevance_lst)

        # k=8
        delf_8_map = get_mAP(delf_8_relevance_lst)

        # k=16
        delf_16_map = get_mAP(delf_16_relevance_lst)

        # Average time
        delf_time_avg = get_average(delf_time_lst)

        # 5) Deep Vision
        # k=2
        dv_2_map = get_mAP(dv_2_relevance_lst)

        # k=4
        dv_4_map = get_mAP(dv_4_relevance_lst)

        # k=8
        dv_8_map = get_mAP(dv_8_relevance_lst)

        # k=16
        dv_16_map = get_mAP(dv_16_relevance_lst)

        # Average time
        dv_time_avg = get_average(dv_time_lst)

        # 6) CROW
        # k=2
        crow_2_map = get_mAP(crow_2_relevance_lst)

        # k=4
        crow_4_map = get_mAP(crow_4_relevance_lst)

        # k=8
        crow_8_map = get_mAP(crow_8_relevance_lst)

        # k=16
        crow_16_map = get_mAP(crow_16_relevance_lst)

        # Average time
        crow_time_avg = get_average(crow_time_lst)

        # 7) CSQ
        # k=2
        csq_2_map = get_mAP(csq_2_relevance_lst)

        # k=4
        csq_4_map = get_mAP(csq_4_relevance_lst)

        # k=8
        csq_8_map = get_mAP(csq_8_relevance_lst)

        # k=16
        csq_16_map = get_mAP(csq_16_relevance_lst)

        # Average time
        csq_time_avg = get_average(csq_time_lst)

        # 8) QIK Show and Tell mAP
        # k=2
        qik_show_and_tell_2_map = get_mAP(qik_show_and_tell_2_relevance_lst)

        # k=4
        qik_show_and_tell_4_map = get_mAP(qik_show_and_tell_4_relevance_lst)

        # k=8
        qik_show_and_tell_8_map = get_mAP(qik_show_and_tell_8_relevance_lst)

        # k=16
        qik_show_and_tell_16_map = get_mAP(qik_show_and_tell_16_relevance_lst)

        # Average time
        qik_show_and_tell_time_avg = get_average(qik_show_and_tell_time_lst)

    # Getting the print string
    output_str = category_combination, qik_2_map, qik_4_map, qik_8_map, qik_16_map, \
                 qik_show_and_tell_2_map, qik_show_and_tell_4_map, qik_show_and_tell_8_map, qik_show_and_tell_16_map, \
                 dir_2_map, dir_4_map, dir_8_map, dir_16_map, \
                 lire_2_map, lire_4_map, lire_8_map, lire_16_map, \
                 delf_2_map, delf_4_map, delf_8_map, delf_16_map, \
                 dv_2_map, dv_4_map, dv_8_map, dv_16_map, \
                 crow_2_map, crow_4_map, crow_8_map, crow_16_map, \
                 csq_2_map, csq_4_map, csq_8_map, csq_16_map, \
                 qik_time_avg, qik_show_and_tell_time_avg, dir_time_avg, lire_time_avg, delf_time_avg, dv_time_avg, crow_time_avg, csq_time_avg, query_lst_len

    # Auditing the results.
    with open(OUTPUT_FILE, 'a+') as f:
        f.write(str(output_str)[1:-1] + "\n")

    return qik_2_map, qik_4_map, qik_8_map, qik_16_map, qik_show_and_tell_2_map, qik_show_and_tell_4_map, qik_show_and_tell_8_map, qik_show_and_tell_16_map, dir_2_map, dir_4_map, dir_8_map, dir_16_map, lire_2_map, lire_4_map, lire_8_map, lire_16_map, delf_2_map, delf_4_map, delf_8_map, delf_16_map, dv_2_map, dv_4_map, dv_8_map, dv_16_map, crow_2_map, crow_4_map, crow_8_map, crow_16_map, csq_2_map, csq_4_map, csq_8_map, csq_16_map, qik_time_avg, qik_show_and_tell_time_avg, dir_time_avg, lire_time_avg, delf_time_avg, dv_time_avg, crow_time_avg, csq_time_avg, query_lst_len


# Ref: https://gist.github.com/bwhite/3726239 - Start
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])
# Ref: https://gist.github.com/bwhite/3726239 - End


def get_mAP(results):
    return mean_average_precision(results)


def get_binary_relevance(results, ground_truth):
    relevance_results = [1 if result in ground_truth else 0 for result in results]
    return relevance_results


# Function to get the average for a list.
def get_average(results):
    if len(results) == 0:
        return 0
    total_average = 0

    for average in results:
        total_average += average

    mean_average = total_average / len(results)
    return mean_average

def get_images(categories):

    # Return list containing all the images.
    image_list = []

    # Get all images containing given categories, select one at random.
    catIds = coco.getCatIds(catNms=categories);
    imgIds = coco.getImgIds(catIds=catIds);

    # Return if there are no images for a particular category combinaion.
    if not imgIds:
        return None

    # Loading the annotations
    imgIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds);
    anns = coco.loadAnns(imgIds)

    for ann in anns:
        img = coco.loadImgs(ann['image_id'])[0]

        if img['file_name'] not in image_subset:
            continue

        if img['file_name'] not in image_list:
            image_list.append(img['file_name'])

    # Return if there are no images for a particular category combinaion.
    if not image_list:
        return None

    return image_list


def get_multicategory_images(image_cat_lst):
    for cat_list in image_cat_lst:
        image_list = get_images(cat_list)
        if image_list is not None:
            return image_list


def eval(category):
    global category_combination
    image_cat_lst = []

    # Check if there are multiple categories
    if "," in category:
        for cat in category.split(","):
            image_cat_lst.append(cat)
    else:
        image_cat_lst = [category]

    category_combination = '_'.join(image_cat_lst)

    # Creating the list of images for the category combination.
    image_cat_list = get_multicategory_images([image_cat_lst])

    if image_cat_list is not None:
        # Starting the evaluation.
        return evaluate(image_cat_list)

    else:
        print("evaluate.py :: eval :: Cannot perform evaluation for the category combination :: ", category_combination)
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def evaluate_cat_comb(category_combination_file):
    # 1) QIK Results List.
    qik_2_mean_average_precision_lst = []
    qik_4_mean_average_precision_lst = []
    qik_8_mean_average_precision_lst = []
    qik_16_mean_average_precision_lst = []
    qik_time_lst = []

    # 2) DIR Results List.
    dir_2_mean_average_precision_lst = []
    dir_4_mean_average_precision_lst = []
    dir_8_mean_average_precision_lst = []
    dir_16_mean_average_precision_lst = []
    dir_time_lst = []

    # 3) LIRE Results List.
    lire_2_mean_average_precision_lst = []
    lire_4_mean_average_precision_lst = []
    lire_8_mean_average_precision_lst = []
    lire_16_mean_average_precision_lst = []
    lire_time_lst = []

    # 4) DELF Results List.
    delf_2_mean_average_precision_lst = []
    delf_4_mean_average_precision_lst = []
    delf_8_mean_average_precision_lst = []
    delf_16_mean_average_precision_lst = []
    delf_time_lst = []

    # 5) Deep Vision Results List.
    dv_2_mean_average_precision_lst = []
    dv_4_mean_average_precision_lst = []
    dv_8_mean_average_precision_lst = []
    dv_16_mean_average_precision_lst = []
    dv_time_lst = []

    # 6) DIR Results List.
    crow_2_mean_average_precision_lst = []
    crow_4_mean_average_precision_lst = []
    crow_8_mean_average_precision_lst = []
    crow_16_mean_average_precision_lst = []
    crow_time_lst = []

    # 7) CSQ Results List.
    csq_2_mean_average_precision_lst = []
    csq_4_mean_average_precision_lst = []
    csq_8_mean_average_precision_lst = []
    csq_16_mean_average_precision_lst = []
    csq_time_lst = []

    # 7) QIK Show and Tell Results List.
    qik_show_and_tell_2_mean_average_precision_lst = []
    qik_show_and_tell_4_mean_average_precision_lst = []
    qik_show_and_tell_8_mean_average_precision_lst = []
    qik_show_and_tell_16_mean_average_precision_lst = []
    qik_show_and_tell_time_lst = []

    # Total queries.
    query_len_lst = []

    # Reading the category combination file.
    f = open(category_combination_file, "r")
    for cat_comb in f:

        # Evaluating with the category combination
        qik_2_map, qik_4_map, qik_8_map, qik_16_map, qik_show_and_tell_2_map, qik_show_and_tell_4_map, qik_show_and_tell_8_map, qik_show_and_tell_16_map, dir_2_map, dir_4_map, dir_8_map, dir_16_map, lire_2_map, lire_4_map, lire_8_map, lire_16_map, delf_2_map, delf_4_map, delf_8_map, delf_16_map, dv_2_map, dv_4_map, dv_8_map, dv_16_map, crow_2_map, crow_4_map, crow_8_map, crow_16_map, csq_2_map, csq_4_map, csq_8_map, csq_16_map, qik_time_avg, qik_show_and_tell_time_avg, dir_time_avg, lire_time_avg, delf_time_avg, dv_time_avg, crow_time_avg, csq_time_avg, query_lst_len = eval(cat_comb.rstrip())

        if qik_2_map is None or not query_lst_len:
            continue

        # Adding QIK results.
        qik_2_mean_average_precision_lst.append(qik_2_map)
        qik_4_mean_average_precision_lst.append(qik_4_map)
        qik_8_mean_average_precision_lst.append(qik_8_map)
        qik_16_mean_average_precision_lst.append(qik_16_map)
        qik_time_lst.append(qik_time_avg)

        # Adding QIK Show and tell results.
        qik_show_and_tell_2_mean_average_precision_lst.append(qik_show_and_tell_2_map)
        qik_show_and_tell_4_mean_average_precision_lst.append(qik_show_and_tell_4_map)
        qik_show_and_tell_8_mean_average_precision_lst.append(qik_show_and_tell_8_map)
        qik_show_and_tell_16_mean_average_precision_lst.append(qik_show_and_tell_16_map)
        qik_show_and_tell_time_lst.append(qik_show_and_tell_time_avg)

        # Adding DIR results.
        dir_2_mean_average_precision_lst.append(dir_2_map)
        dir_4_mean_average_precision_lst.append(dir_4_map)
        dir_8_mean_average_precision_lst.append(dir_8_map)
        dir_16_mean_average_precision_lst.append(dir_16_map)
        dir_time_lst.append(dir_time_avg)

        # Adding LIRE results.
        lire_2_mean_average_precision_lst.append(lire_2_map)
        lire_4_mean_average_precision_lst.append(lire_4_map)
        lire_8_mean_average_precision_lst.append(lire_8_map)
        lire_16_mean_average_precision_lst.append(lire_16_map)
        lire_time_lst.append(lire_time_avg)

        # Adding DELF results.
        delf_2_mean_average_precision_lst.append(delf_2_map)
        delf_4_mean_average_precision_lst.append(delf_4_map)
        delf_8_mean_average_precision_lst.append(delf_8_map)
        delf_16_mean_average_precision_lst.append(delf_16_map)
        delf_time_lst.append(delf_time_avg)

        # Adding DeepVision results.
        dv_2_mean_average_precision_lst.append(dv_2_map)
        dv_4_mean_average_precision_lst.append(dv_4_map)
        dv_8_mean_average_precision_lst.append(dv_8_map)
        dv_16_mean_average_precision_lst.append(dv_16_map)
        dv_time_lst.append(dv_time_avg)

        # Adding CroW results.
        crow_2_mean_average_precision_lst.append(crow_2_map)
        crow_4_mean_average_precision_lst.append(crow_4_map)
        crow_8_mean_average_precision_lst.append(crow_8_map)
        crow_16_mean_average_precision_lst.append(crow_16_map)
        crow_time_lst.append(crow_time_avg)

        # Adding CSQ results.
        csq_2_mean_average_precision_lst.append(csq_2_map)
        csq_4_mean_average_precision_lst.append(csq_4_map)
        csq_8_mean_average_precision_lst.append(csq_8_map)
        csq_16_mean_average_precision_lst.append(csq_16_map)
        csq_time_lst.append(csq_time_avg)

        # Adding the query length.
        query_len_lst.append(query_lst_len)

    # QIK
    qik_2_average = get_average(qik_2_mean_average_precision_lst)
    qik_4_average = get_average(qik_4_mean_average_precision_lst)
    qik_8_average = get_average(qik_8_mean_average_precision_lst)
    qik_16_average = get_average(qik_16_mean_average_precision_lst)
    qik_time_average = get_average(qik_time_lst)

    # QIK Show and Tell
    qik_show_and_tell_2_average = get_average(qik_show_and_tell_2_mean_average_precision_lst)
    qik_show_and_tell_4_average = get_average(qik_show_and_tell_4_mean_average_precision_lst)
    qik_show_and_tell_8_average = get_average(qik_show_and_tell_8_mean_average_precision_lst)
    qik_show_and_tell_16_average = get_average(qik_show_and_tell_16_mean_average_precision_lst)
    qik_show_and_tell_time_average = get_average(qik_show_and_tell_time_lst)

    # DIR
    dir_2_average = get_average(dir_2_mean_average_precision_lst)
    dir_4_average = get_average(dir_4_mean_average_precision_lst)
    dir_8_average = get_average(dir_8_mean_average_precision_lst)
    dir_16_average = get_average(dir_16_mean_average_precision_lst)
    dir_time_average = get_average(dir_time_lst)

    # LIRE
    lire_2_average = get_average(lire_2_mean_average_precision_lst)
    lire_4_average = get_average(lire_4_mean_average_precision_lst)
    lire_8_average = get_average(lire_8_mean_average_precision_lst)
    lire_16_average = get_average(lire_16_mean_average_precision_lst)
    lire_time_average = get_average(lire_time_lst)

    # DELF
    delf_2_average = get_average(delf_2_mean_average_precision_lst)
    delf_4_average = get_average(delf_4_mean_average_precision_lst)
    delf_8_average = get_average(delf_8_mean_average_precision_lst)
    delf_16_average = get_average(delf_16_mean_average_precision_lst)
    delf_time_average = get_average(delf_time_lst)

    # DeepVision
    dv_2_average = get_average(dv_2_mean_average_precision_lst)
    dv_4_average = get_average(dv_4_mean_average_precision_lst)
    dv_8_average = get_average(dv_8_mean_average_precision_lst)
    dv_16_average = get_average(dv_16_mean_average_precision_lst)
    dv_time_average = get_average(dv_time_lst)

    # CroW
    crow_2_average = get_average(crow_2_mean_average_precision_lst)
    crow_4_average = get_average(crow_4_mean_average_precision_lst)
    crow_8_average = get_average(crow_8_mean_average_precision_lst)
    crow_16_average = get_average(crow_16_mean_average_precision_lst)
    crow_time_average = get_average(crow_time_lst)

    # CSQ
    csq_2_average = get_average(csq_2_mean_average_precision_lst)
    csq_4_average = get_average(csq_4_mean_average_precision_lst)
    csq_8_average = get_average(csq_8_mean_average_precision_lst)
    csq_16_average = get_average(csq_16_mean_average_precision_lst)
    csq_time_average = get_average(csq_time_lst)

    # Summing query lengths.
    sum_query = sum(query_len_lst)

    # Pretty printing the results.
    t = PrettyTable(['System', 'k=2', 'k=4', 'k=8', 'k=16', "Average Time(s)"])
    t.add_row(['QIK', round(qik_2_average, 2), round(qik_4_average, 2), round(qik_8_average, 2), round(qik_16_average, 2), round(qik_time_average, 2)])
    t.add_row(['QIK (Show &v Tell)', round(qik_show_and_tell_2_average, 2), round(qik_show_and_tell_4_average, 2), round(qik_show_and_tell_8_average, 2), round(qik_show_and_tell_16_average, 2), round(qik_show_and_tell_time_average, 2)])
    t.add_row(['CSQ', round(csq_2_average, 2), round(csq_4_average, 2), round(csq_8_average, 2), round(csq_16_average, 2), round(csq_time_average, 2)])
    t.add_row(['CroW', round(crow_2_average, 2), round(crow_4_average, 2), round(crow_8_average, 2), round(crow_16_average, 2), round(crow_time_average, 2)])
    t.add_row(['FR-CNN', round(dv_2_average, 2), round(dv_4_average, 2), round(dv_8_average, 2), round(dv_16_average, 2), round(dv_time_average, 2)])
    t.add_row(['DIR', round(dir_2_average, 2), round(dir_4_average, 2), round(dir_8_average, 2), round(dir_16_average, 2), round(dir_time_average, 2)])
    t.add_row(['DELF', round(delf_2_average, 2), round(delf_4_average, 2), round(delf_8_average, 2), round(delf_16_average, 2), round(delf_time_average, 2)])
    t.add_row(['LIRE', round(lire_2_average, 2), round(lire_4_average, 2), round(lire_8_average, 2), round(lire_16_average, 2), round(lire_time_average, 2)])
    print(t)
    print("Total no of queries considered = ", sum_query)

if __name__ == '__main__':
    # Setting the global variables with user input.
    parser = argparse.ArgumentParser(description='Compute MAP for pre-fetched query results.')
    parser.add_argument('-image_data', default="data/15K_Dataset.pkl", metavar='data', help='Pickled file containing the list of images.', required=False)
    parser.add_argument('-threshold', default=".70", type=float, help='Sentence similarity threshold.', required=False)
    parser.add_argument('-pre_computed_results', default="pre_constructed_data/15K_Results.pkl", help='Pre-fetched results file path.', required=False)
    parser.add_argument('-ground_truth', default="data/Ground_Truth.pkl", help='Pre-constructed ground truth.', required=False)
    parser.add_argument('-categories', default="data/2_cat_comb.txt", help='Category combination input file path.', required=False)
    parser.add_argument('-outfile', default="data/QIK_Output_Combined.txt",help='MAP output file path.', required=False)
    args = parser.parse_args()

    IMAGE_SET_PATH = args.image_data
    SIMILARITY_THRESHOLD = args.threshold
    PRE_COMPUTED_RESULTS_PATH = args.pre_computed_results
    OUTPUT_FILE = args.outfile
    PRE_COMPUTED_GROUND_TRUTH_PATH = args.ground_truth

    # Read the annotation files.
    init()

    # Performing the evaluation
    evaluate_cat_comb(args.categories)
