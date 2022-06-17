# imports
import pickle
import argparse
import json

# Function to get the average for a list
def get_average(results):
    if len(results) == 0:
        return 0
    total_average = 0

    for average in results:
        total_average += average

    mean_average = total_average / len(results)
    return mean_average

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Average Execution time')
    parser.add_argument('-data', default="pre_constructed_data/QIK_ClipCap_Evaluation_MSCOCO_120k.txt", metavar='data', help='Data File', required=False)
    parser.add_argument('-technique', default="qik", metavar='data', help='Technique', required=False)
    args = parser.parse_args()

    # Iterating over the data to create a list of times
    time_lst = []
    results = open(args.data, "r")
    for result in results:
        res_json = json.loads(result.split("::")[1].replace("'",'"'))
        time_lst.append(res_json[args.technique + "_time"])

    # Printing the average time taken
    print("Average Time Taken: ", get_average(time_lst))