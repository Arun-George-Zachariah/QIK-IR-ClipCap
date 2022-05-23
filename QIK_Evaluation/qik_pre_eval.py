from sys import path
path.append("../QIK_Web/util/")
path.append("../ML_Models/ObjectDetection")

import constants
from qik_search import qik_search
import datetime
import clipcap_caption_generator


def retrieve(query_image):
    ret_dict = {}

    # Reading the input request.
    query_image_path = constants.TOMCAT_LOC + constants.IMAGE_DATA_DIR + query_image

    # Get QIK results
    time = datetime.datetime.now()

    qik_pre_results = []
    qik_results = []

    # Fetching the candidates from QIK.
    try:
        query, qik_results_dict, similar_images = qik_search(query_image_path, obj_det_enabled=False, ranking_func='Parse Tree', fetch_count=None, is_similar_search_enabled=False)
    except:
        print("qik_pre_eval :: Exception encountered for the query image :: ", query_image_path)
        return None

    for result in qik_results_dict:
        k, v = result
        qik_pre_results.append(k.split("::")[0].split("/")[-1])

    # Noting QIK time.
    qik_time = datetime.datetime.now() - time

    # Removing query image from the result set.
    for res in qik_pre_results:
        if res == query_image:
            continue
        qik_results.append(res)

    # Adding data to the return dictionary.
    ret_dict["qik_time"] = qik_time.total_seconds()
    ret_dict["qik_results"] = qik_results

    # Writing the output to a file.
    with open("data/QIK_Results_Dict.txt", 'a+') as f:
        f.write(query_image + ":: " + str(ret_dict) + "\n")

    print("qik_pre_eval :: retrieve :: ret_dict :: ", str(ret_dict))
    return ret_dict


if __name__ == '__main__':

    # Initial Loading of the caption generator model.
    clipcap_caption_generator.init()

    # Reading the images from the file.
    images = open("data/Images.txt", "r")
    for image in images:
        image = image.rstrip()
        print("qik_pre_eval :: Executing :: ", image)

        res = retrieve(image)
        print("qik_pre_eval :: Process :: res :: ", image, ",", res)
