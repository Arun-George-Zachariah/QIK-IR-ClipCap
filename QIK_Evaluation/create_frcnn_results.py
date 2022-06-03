from sys import path
path.append("../QIK_Web/util/")
path.append("../ML_Models/DeepVision")

import constants
import logging
import datetime
import argparse
import pickle
import frcnn_search

# Local constants
EVAL_K = 16

def retrieve(query_image):
    ret_dict = {}

    # Reading the input request.
    query_image_path = constants.TOMCAT_LOC + constants.IMAGE_DATA_DIR + query_image

    # Get CroW results
    time = datetime.datetime.now()

    frcnn_results = []

    # Fetching the candidates from CroW.
    frcnn_pre_results = frcnn_search.frcnn_search(query_image_path, EVAL_K + 1)

    # Noting CroW time.
    frcnn_time = datetime.datetime.now() - time
    print("create_frcnn_results.py :: CroW Fetch Execution time :: ", frcnn_time)
    logging.info("create_frcnn_results.py :: CroW Fetch Execution time :: %s", str(frcnn_time))

    # Removing query image from the result set.
    for res in frcnn_pre_results:
        img_file = res.rstrip().split("/")[-1]
        if img_file == query_image:
            continue
        frcnn_results.append(res.rstrip().split("/")[-1])
    print("create_frcnn_results.py :: CroW :: frcnn_results :: ", frcnn_results)

    # Adding data to the return dictionary.
    ret_dict["frcnn_time"] = frcnn_time.total_seconds()
    ret_dict["frcnn_results"] = frcnn_results

    # Writing the output to a file.
    with open("data/CroW_Results_Dict.txt", 'a+') as f:
        f.write(query_image + ":: " + str(ret_dict) + "\n")

    print("create_frcnn_results.py :: retrieve :: ret_dict :: ", str(ret_dict))
    return ret_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess to extract text from a folder of csv files.')
    parser.add_argument('-data', default="data/15K_Dataset.pkl", metavar='data', help='Pickled file containing the list of images.', required=False)
    parser.add_argument('-out', default="data/CroW_Results.pkl", metavar='data', help='Directory to write the CroW Results.', required=False)
    args = parser.parse_args()

    # Initializing CroW.
    frcnn_search.init()

    # Initializing a dictionary to hold the results.
    results_dict = {}

    # Loading the image set.
    image_subset = pickle.load(open(args.data, "rb"))

    # Iterating over the image set.
    for image in image_subset:
        results_dict[image] = retrieve(image)

    # Creating the pickle file of the ground truth.
    with open(args.out, "wb") as f:
        pickle.dump(results_dict, f)
