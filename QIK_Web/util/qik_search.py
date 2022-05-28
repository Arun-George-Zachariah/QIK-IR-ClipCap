from sys import path
path.append("../APTED/apted")
path.append("../ML_Models/ObjectDetection")
path.append("util/")

import clipcap_caption_generator
import constants
import json
import requests
import datetime
import parse_show_tree
import urllib
from apted import APTED, PerEditOperationConfig
import apted.helpers as apth
import detect_objects
import random
from concurrent.futures import ThreadPoolExecutor


def rank_candidates(query_rep, cand_rep, captionRanksDict, key):
    parseTED = APTED(apth.Tree.from_text(query_rep), apth.Tree.from_text(cand_rep),
                                 PerEditOperationConfig(1, 1, 1)).compute_edit_distance()

    captionRanksDict[key] = parseTED


def get_similar_images(query_image):
    similar_images = {}

    # Noting the time taken for further auditing.
    time = datetime.datetime.now()

    # Querying the backend to fetch the list of similar images.
    cap_req = constants.SIMILAR_SEARCH_URL + query_image
    cap_res_text = requests.get(cap_req).text

    try:
        # Converting the response to a JSON.
        cap_res = json.loads(requests.get(cap_req).text)

        # Shuffling the dictionary.
        items = list(cap_res.items())
        random.shuffle(items)
        shuffled_cap_res = dict(items)

        # Replacing the image URL with an updated URL.
        for link in shuffled_cap_res:
            caption = shuffled_cap_res[link]
            updated_link = link.replace(constants.TOMCAT_OLD_IP_ADDR, constants.TOMCAT_IP_ADDR)
            similar_images[updated_link] = caption

        # Auditing the response time.
        # print("qik_search :: get_similar_images :: Execution time :: ", (datetime.datetime.now() - time))
    except:
        print("qik_search :: get_similar_images :: No similar images found.")

    return similar_images

def qik_search(query_image, ranking_func=None, obj_det_enabled=False, pure_objects_search=False, fetch_count=None, is_similar_search_enabled=True):
    obj_res = None
    cap_res = None
    similar_images = None

    captionRanksDict = {}
    sortedCaptionRanksDict = {}

    # Noting the time taken for further auditing.
    time = datetime.datetime.now()

    if obj_det_enabled:
        # Initial Loading of the object detection model.
        detect_objects.init()

        # Detecting objects.
        json_data = {}
        json_data['objects'] = detect_objects.get_detected_objects(query_image, constants.OBJECT_DETECTED_THRESHOLD)

        # Querying the backend to fetch the list of images and captions based on the objects detected.
        obj_req = constants.DETECT_OBJECTS_URL + urllib.parse.quote(str(json_data))
        obj_res = json.loads(requests.get(obj_req).text)

    if pure_objects_search:
        if obj_res is not None:
            # Forming the return image set.
            for resMap in obj_res:
                caption = resMap['caption']
                image = resMap['fileURL']

                # Temp Fix done to replace Tomcat IP. Needs to be handled in the IndexEngine.
                image_path = image.replace(constants.TOMCAT_OLD_IP_ADDR, constants.TOMCAT_IP_ADDR)

                captionRanksDict[image_path + ":: " + caption] = 1

            # Formating done for Ranking
            sortedCaptionRanksDict = sorted(captionRanksDict.items(), key=lambda kv: kv[1], reverse=True)

            # Auditing the QIK execution time.
            # print("QIK Execution time :: ", (datetime.datetime.now() - time))

            if sortedCaptionRanksDict and fetch_count is not None:
                return "Query Image", sortedCaptionRanksDict[:fetch_count], None
            else:
                return  "Query Image", sortedCaptionRanksDict, None

        return "Query Image", sortedCaptionRanksDict, None

    # Initial Loading of the caption generator model.
    clipcap_caption_generator.init()

    # Generating the captions.
    query = clipcap_caption_generator.get_captions(query_image)

    # Handling the fullstops in captions.
    if query[-1] == '.':
        query = query[:-1].strip()

    # Querying the backend to fetch the list of images and captions.
    cap_req = constants.SOLR_QUERY_URL + query
    cap_res = requests.get(cap_req).text
    if cap_res is not None:
        cap_res = json.loads(cap_res)
    # print("QIK Fetch Execution time :: ", (datetime.datetime.now() - time))

    # Merging the two responses.
    if obj_res is None:
        res = cap_res
    elif cap_res is None:
        res = obj_res
    else:
        res = obj_res + cap_res
    
    # Forming the return image set.
    if res is not None:
        # Performing TED based Ranking on the parse tree.
        if ranking_func == 'Parse Tree':

            # Generating the parse tree for the input query.
            queryParseTree = parse_show_tree.parseSentence(query)

            with ThreadPoolExecutor(max_workers=40) as exe:
                for resMap in res:
                    image = resMap['fileURL']
                    caption = resMap['caption']
                    captionParseTree = resMap['parseTree']

                    # Temp Fix done to replace Tomcat IP. Needs to be handled in the IndexEngine.
                    image_path = image.replace(constants.TOMCAT_OLD_IP_ADDR, constants.TOMCAT_IP_ADDR)

                    # Ranking the candidates
                    exe.submit(rank_candidates, queryParseTree, captionParseTree, captionRanksDict, image_path + ":: " + caption)

            # Sorting the results based on the Parse TED.
            sortedCaptionRanksDict = sorted(captionRanksDict.items(), key=lambda kv: kv[1], reverse=False)
        
        elif ranking_func == 'Dependency Tree':

            # Generating the dependency tree for the input query.
            queryDepTree = parse_show_tree.dependencyParser(query)

            with ThreadPoolExecutor(max_workers=40) as exe:
                for resMap in res:
                    image = resMap['fileURL']
                    caption = resMap['caption']
                    depTree = resMap['depTree']

                    # Temp Fix done to replace Tomcat IP. Needs to be handled in the IndexEngine.
                    image_path = image.replace(constants.TOMCAT_OLD_IP_ADDR, constants.TOMCAT_IP_ADDR)

                    # Ranking the candidates
                    exe.submit(rank_candidates, queryDepTree, depTree, captionRanksDict, image_path + ":: " + caption)

            # Sorting the results based on the Parse TED.
            sortedCaptionRanksDict = sorted(captionRanksDict.items(), key=lambda kv: kv[1], reverse=False)

        else:
            # Forming the return image set (Without ranking)
            for resMap in res:
                caption = resMap['caption']
                image = resMap['fileURL']

                # Temp Fix done to replace Tomcat IP. Needs to be handled in the IndexEngine.
                image_path = image.replace(constants.TOMCAT_OLD_IP_ADDR, constants.TOMCAT_IP_ADDR)

                captionRanksDict[image_path + ":: " + caption] = 1

            # Formating done for Ranking
            sortedCaptionRanksDict = sorted(captionRanksDict.items(), key=lambda kv: kv[1], reverse=True)

        if is_similar_search_enabled:
            similar_images = get_similar_images(query)

    # Auditing the QIK execution time.
    # print("QIK Execution time :: ", (datetime.datetime.now() - time))

    if sortedCaptionRanksDict and fetch_count is not None:
        return query, sortedCaptionRanksDict[:fetch_count], similar_images
    else:
        return query, sortedCaptionRanksDict, similar_images
