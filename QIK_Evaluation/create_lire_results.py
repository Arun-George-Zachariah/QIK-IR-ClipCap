from sys import path
path.append("../QIK_Web/util/")

import constants
import requests
import datetime
import pickle
import requests
import argparse

# Local constants (To be moved to eval constants)
eval_k = 8

def retrieve(query_image):
    ret_dict = {}

    # Reading the input request.
    query_image_path = constants.TOMCAT_LOC + constants.IMAGE_DATA_DIR + query_image

    # Get LIRE results
    time = datetime.datetime.now()

    lire_pre_results = []
    lire_results = []

    # Fetching the candidates from LIRE.
    req = constants.LIRE_QUERY + query_image_path
    resp = requests.get(req).text[1:-2].split(",")
    for img in resp:
        lire_pre_results.append(img.strip())

    # Noting LIRE time.
    lire_time = datetime.datetime.now() - time

    # Removing query image from the result set.
    for res in lire_pre_results:
        img_file = res.rstrip().split("/")[-1]
        if img_file == query_image:
            continue
        lire_results.append(res.rstrip().split("/")[-1])

    # Adding data to the return dictionary.
    ret_dict["lire_time"] = lire_time.total_seconds()
    ret_dict["lire_results"] = lire_results

    # Writing the output to a file.
    with open("data/LIRE_Pre_Results_Dict.txt", 'a+') as f:
        f.write(query_image + ":: " + str(ret_dict) + "\n")

    print("lire_pre_eval :: retrieve :: ret_dict :: ", str(ret_dict))
    return ret_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess to extract text from a folder of csv files.')
    parser.add_argument('-data', default="data/15K_Dataset.pkl", metavar='data', help='Pickled file containing the list of images.', required=False)
    parser.add_argument('-db_dir', default="/mydata/apache-tomcat/webapps/QIK_Image_Data", metavar='data', help='Directory containing the candidate images.', required=False)
    parser.add_argument('-out', default="data/LIRe_Results.pkl", metavar='data', help='Directory to write the LIRe Results.', required=False)
    args = parser.parse_args()

    # Indexing images
    req_params = {"dir": args.db_dir}
    requests.get(url = "http://localhost:8080/indexLire", params=req_params)

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