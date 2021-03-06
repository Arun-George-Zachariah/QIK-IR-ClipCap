from sys import path
path.append("../ML_Models/ObjectDetection")
path.append("../QIK_Web/util")

import os
import threading
import re
import urllib
import requests
import time
from threading import Thread, Lock
import constants, clipcap_caption_generator
from get_coordinates import ImageMetaData
from nltk.parse.stanford import StanfordDependencyParser
import detect_objects

queue = []
lock = Lock()
threadLimiter = threading.BoundedSemaphore(constants.THREAD_COUNT)
depParser = StanfordDependencyParser()

def print_sentence(sent_list):
    for line in sent_list:
        return line
        
def get_dep_tree(caption):
    dep_sentence = [parse.tree() for parse in depParser.raw_parse(caption)]
    sent = print_sentence(dep_sentence)
    dep_tree = str(sent).replace("\n","").replace("  ","").replace(" (","{").replace("(","{").replace(")","}").replace(" ","{").replace("}{","}}{") + "}"
    return dep_tree
	
# Producer Thread.
class Producer(Thread):
    def run(self):
        global queue

        if os.path.exists(constants.IMAGE_LIST_FILE):
            # Iterating over the images mentioned in the file.
            images = open(constants.IMAGE_LIST_FILE, "r")
            for image in images:
                # Acquiring the lock before adding the to the queue.
                lock.acquire()

                # Adding the file to the queue.
                queue.append(os.fsdecode(image.rstrip()))

                # Releasing the lock acquired.
                lock.release()


        else:
            # Iterating over the images in the directory.
            for file in os.listdir(os.fsencode(constants.IMAGE_DIR)):
                # Acquiring the lock before adding the to the queue.
                lock.acquire()

                # Adding the file to the queue.
                queue.append(os.fsdecode(file))

                # Releasing the lock acquired.
                lock.release()

# Consumer Thread.
class Consumer(Thread):
    def run(self):
        global queue
        while True:
            if not queue:
                #Nothing in queue, but consumer will try to consume
                continue

            # Acquiring the lock before removing from the queue.
            lock.acquire()

            # Fetching the files from the queue.
            file = queue.pop(0)

            # Releasing the lock acquired.
            lock.release()

            # Instantiate a thread with the file name as the thread name.
            process = Process(name=file)
            process.start()
            time.sleep(1)

class Process(threading.Thread):
    def run(self):
        threadLimiter.acquire()
        try:
            self.exec()
        finally:
            threadLimiter.release()

    def exec(self):
        # Getting the complete image path
        filename = constants.IMAGE_DIR + self.getName()
        print("Starting :: ", filename)
        
        # Fetching the geo co-ordinates for the images.
        meta_data =  ImageMetaData(filename)
        latlng = meta_data.get_lat_lng()

        # Forming a json data from the captions generated.
        json_data = {}
        
        # Adding the geo co-ordinates to the meta data
        if latlng[0] is not None:
            json_data['lat'] = latlng[0]
        if latlng[1] is not None:
            json_data['lng'] = latlng[1]

        # Adding objects detected to the metedata
        json_data['objects'] = detect_objects.get_detected_objects(filename)
        json_data['key'] = constants.IMAGE_URL + os.fsdecode(self.getName())

        # Fetching the captions for the images.
        caption_str = clipcap_caption_generator.get_captions(filename)
        json_data['cap1_cap'] = caption_str

        # Constructing the dependency tree for the caption.
        json_data['cap1_dep_tree'] = get_dep_tree(caption_str)
                    
        print(json_data)

        # Posting the captions to the index engine.
        req = constants.INDEX_ENGINE_URL + urllib.parse.quote(str(json_data))
        requests.get(req)

        print("Finished :: ", self.getName())

if __name__ == "__main__":
    # Loading the object detection model.
    detect_objects.init()

    # Loading the ClipCap Model.
    clipcap_caption_generator.init()
    
    # Starting the producer process
    Producer().start()

    # Starting the consumer process.
    Consumer().start()
