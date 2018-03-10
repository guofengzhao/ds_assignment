# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:50:56 2018

@author: Guofeng.Zhao

This assignment involve Record Deduplicate and Record Linkage problems. 
Since there is no labelled data to train a ML model, I opted to implement this 
task using the very basic tools available in Python, NumPy and Pandas.

At a high level, I am taking the following approach -

1. Load the DBLP csv file into a DataFrame
2. Clean and normalize the DBLP dataset
3. Deduplicate the DBLP dataset using DataFrame, doable due to the data size

4. Load the Scholar csv file into a DataFrame
5. Clean and normalize the Scholar dataset
6. Optionally, deduplicate the Scholar dataset using NumPy Array utility. 
DataFrame won't do the job because of the data size. This step is not 
absolutely necessary because I will seek record linkages from DBLP to Scholar, 
the right-hand side dataset with duplicates really does not matter.

7. Iterate through the DBLP records and seek for identical entity in the 
Scholar records.
8. Write the found matches to the output csv file.

Note: For both deduplication and linkage resolution, the key is to calculate 
the similarity between two given records. I implemented the necessary helper 
functions in this assignment, using Ratcliff/Obershelp(difflib), Levenshtein, 
Jaccard, and Sorensen distance methods. Experiments shows the 
Ratcliff/Obershelp algorithm performs faster and gets the similar results.

The similarities are evaluated on the column "title", "authors", "venue" and 
"year". The similarity thresholds on these columns along with the default 
distance measure method are all tunable at the begining of the code.

The packages being used -

* Python 3.6
* NumPy
* Pandas
* Distance

This is the first step of the phased processing.
     
"""

# Import the required packages
import pandas as pd
import numpy as np
import re
import difflib
import distance
from datetime import datetime

# Tunable parameters

# input and output file names
"""
fname_dblp = "../inputs/DBLP1000.csv"
fname_scholar = "../inputs/Scholar2000.csv"
fname_result = "../outputs/DBLP1000_Scholar2000_perfectMapping_GuofengZhao.csv"
fname_dblp_clean = "../outputs/DBLP1000_clean.csv"
fname_dblp_dupe = "../outputs/DBLP1000_dupe.csv"
fname_scholar_clean = "../outputs/Scholar2000_clean.csv"
fname_scholar_dupe = "../outputs/Scholar2000_dupe.csv"
"""

fname_dblp = "../inputs/DBLP1.csv"
fname_scholar = "../inputs/Scholar.csv"
fname_result = "../outputs/DBLP_Scholar_perfectMapping_GuofengZhao.csv"
fname_dblp_clean = "../outputs/DBLP_clean.csv"
fname_dblp_dupe = "../outputs/DBLP_dupe.csv"
fname_scholar_clean = "../outputs/Scholar_clean.csv"
fname_scholar_dupe = "../outputs/Scholar_dupe.csv"

# similarity thresholds
title_similarity_threshold = 0.90
authors_similarity_threshold = 0.50
venue_similarity_threshold = 0.90
year_similarity_threshold = 0.99

# default distance method, one of 'difflib', 'levenshtein', 'sorensen', or 'jaccard'
default_distance = 'difflib'

# generic helper functions

def load_prepare_dataset (fname):
    dataset = pd.read_csv(fname, quotechar='"', encoding='ansi', engine='python')
    print ("DataFrame demensions: {}".format(dataset.shape))
    print (dataset.count())
    # change to lower case and trim the leading and trailing spaces
    dataset['title'] = dataset['title'].map(lambda x: x if type(x)!=str else x.lower().strip())
    dataset['authors'] = dataset['authors'].map(lambda x: x if type(x)!=str else x.lower().strip())
    dataset['venue'] = dataset['venue'].map(lambda x: x if type(x)!=str else x.lower().strip())
    return dataset

# helper functions for the dataframe 
def title_similarity_pd(row, method = 'difflib'):
    if method.lower() == "levenshtein":
        return 1- distance.nlevenshtein(row["title"], row["title_R"], method=1)
    if method.lower() == "sorensen":
        return 1- distance.sorensen(row["title"], row["title_R"])
    if method.lower() == "jaccard":
        return 1- distance.jaccard(row["title"], row["title_R"])
    return difflib.SequenceMatcher(None, row["title"], row["title_R"]).quick_ratio()
    
def authors_similarity_pd (row):
    if pd.isnull(row["authors"]):
        return 1
    if pd.isnull(row["authors_R"]):
        return 1
    return difflib.SequenceMatcher(None, re.split(r'[;,\s]\s*', row["authors"]), re.split(r'[;,\s]\s*', row["authors_R"])).quick_ratio()

def venue_similarity_pd (row):
    if pd.isnull(row["venue"]):
        return 1
    if pd.isnull(row["venue_R"]):
        return 1
    return difflib.SequenceMatcher(None, row["venue"], row["venue_R"]).quick_ratio()

def year_similarity_pd (row):
    if pd.isnull(row["year"]):
        return 1
    if pd.isnull(row["year_R"]):
        return 1
    return 1 if row["year"] == row["year_R"] else 0

def row_similar_pd (row):
    if title_similarity_pd (row, method = default_distance) < title_similarity_threshold:
        return False
    if authors_similarity_pd (row) < authors_similarity_threshold:
        return False
    if venue_similarity_pd (row) < venue_similarity_threshold:
        return False
    if year_similarity_pd (row) < year_similarity_threshold:
        return False
    return True

# Deduplicate using DataFrame approach
def deduplicate_pd (dataset):
    # generate the N*N pair matrix
    dataset['tmp'] = 1
    max_row_id = dataset.loc[:, ["Row_ID"]].max().at["Row_ID"]
    dataset_pair = dataset.merge(dataset, on='tmp', suffixes=["","_R"])
    print("Total pairs after initial merge: {}".format(dataset_pair.shape))
    #dataset_pair.drop(labels=['tmp'], axis=1, inplace=True)
    
    # half fold to remove duplicate pairs
    dataset_pair = dataset_pair[(dataset_pair["Row_ID"] < dataset_pair["Row_ID_R"]) | (dataset_pair["Row_ID_R"] == max_row_id)]
    print("Total pairs after half-fold deduplicate: {}".format(dataset_pair.shape))
    
    # calculate and populate similarity measurements to all pairs
    print("Start calculating similarity - " + str(datetime.now()))
    dataset_pair['row_similar'] = dataset_pair.apply(row_similar_pd, axis = 1)
    print("Finished calculating similarity - " + str(datetime.now()))
    print("Total pairs after similarity evaluation: {}".format(dataset_pair.shape))

    x = dataset_pair.groupby(by="Row_ID", as_index=False).agg("max")
    dupe_rows = x[x["row_similar"] == True]
    dupe_rows = dupe_rows.loc[:, ['idDBLP', 'title', 'authors', 'venue', 'year', 'Row_ID']]
    print("similar pairs: {}".format(dupe_rows.shape))
    
    unique_rows = x[x["row_similar"] == False]
    unique_rows = unique_rows.loc[:, ['idDBLP', 'title', 'authors', 'venue', 'year', 'Row_ID']]
    print("distinct pairs: {}".format(unique_rows.shape))
    
    return dupe_rows, unique_rows

# helper functions for the numpy array
def title_similarity_np(row1, row2, method = "difflib"):
    if method.lower() == "levenshtein":
        return 1- distance.nlevenshtein(row1[1], row2[1], method=1)
    if method.lower() == "sorensen":
        return 1- distance.sorensen(row1[1], row2[1])
    if method.lower() == "jaccard":
        return 1- distance.jaccard(row1[1], row2[1])
    return difflib.SequenceMatcher(None, row1[1], row2[1]).quick_ratio()
    
def authors_similarity_np (row1, row2):
    if pd.isnull(row1[2]):
        return 1
    if pd.isnull(row2[2]):
        return 1
    return difflib.SequenceMatcher(None, re.split(r'[;,\s]\s*', row1[2]), re.split(r'[;,\s]\s*', row2[2])).quick_ratio()
def venue_similarity_np (row1, row2):
    if pd.isnull(row1[3]):
        return 1
    if pd.isnull(row2[3]):
        return 1
    return difflib.SequenceMatcher(None, row1[3], row2[3]).quick_ratio()

def year_similarity_np (row1, row2):
    if pd.isnull(row1[4]):
        return 1
    if pd.isnull(row2[4]):
        return 1
    return 1 if row1[4] == row2[4] else 0

def row_similar_np (row1, row2):
    if title_similarity_np (row1, row2, method = default_distance) < title_similarity_threshold:
        return False
    if authors_similarity_np (row1, row2) < authors_similarity_threshold:
        return False
    if venue_similarity_np (row1, row2) < venue_similarity_threshold:
        return False
    if year_similarity_np (row1, row2) < year_similarity_threshold:
        return False
    return True

# Deduplicate using NumPy Array approach
def deduplicate_np (dataset):
    dataset['duplicate'] = -1
    data_columns = dataset.axes[1].values
    data_values = dataset.values
    nrows, ncolumns = data_values.shape
    print("Start calculating similarity - " + str(datetime.now()))
    count = 0
    for i in range(nrows):
        for j in range(i+1, nrows):
            count = count+1
            if count % 100000 == 0:
                print("{} comparison calculated.".format(count))
            if row_similar_np(data_values[i, :], data_values[j, :]):
                data_values[i, 6] = data_values[j, 5]
                if pd.isnull(data_values[j, 2]):
                    data_values[j, 2] = data_values[i, 2]
                if pd.isnull(data_values[j, 3]):
                    data_values[j, 3] = data_values[i, 3]
                if pd.isnull(data_values[j, 2]):
                    data_values[j, 4] = data_values[i, 4]
                break
    print("{} comparison calculated.".format(count))
    print("Finished calculating similarity - " + str(datetime.now()))
    x = pd.DataFrame(data = data_values, columns = data_columns)
    dupe_rows = x[x["duplicate"] >= 0]
    print("duplicates: {}".format(dupe_rows.shape))
    unique_rows = x[x["duplicate"] < 0]
    print("distinct rows: {}".format(unique_rows.shape))
    
    return dupe_rows, unique_rows 

# function to seek for matches between two datasets
def link_records (dataset1, dataset2):
    links_values = []
    links_columns = ['idDBLP', 'idScholar', 'DBLP_Match', 'Scholar_Match', 'Match_ID']
    data1_values = dataset1.values
    data2_values = dataset2.values
    nrows1 = data1_values.shape[0]
    nrows2 = data2_values.shape[0]

    print("Start seeking for matches - " + str(datetime.now()))
    checks = 0
    matches = 0
    for i in range(nrows1):
        for j in range(nrows2):
            checks = checks+1
            if checks % 100000 == 0:
                print("{} comparison calculated.".format(checks))
            if row_similar_np (data1_values[i, :], data2_values[j, :]):
                a_link = [data1_values[i, 0], data2_values[j, 0], data1_values[i, 5], data2_values[j, 5], str(data1_values[i, 5])+'_'+str(data2_values[j, 5])]
                links_values.append(a_link)
                matches = matches + 1
                if matches % 100 == 0:
                    print("{} matches found.".format(matches))
                break    
    print("{} comparison calculated.".format(checks))
    print("Finished seeking for matches - " + str(datetime.now()))
    print("TOTALLY {} matches found.".format(matches))
    links = pd.DataFrame(data = links_values, columns = links_columns)
    
    return links

"""
 ===== MAIN FLOW STARTS FROM HERE =====
 
"""
# load and prepare the DBLP dataset

dblp = load_prepare_dataset(fname_dblp)

# deduplicate the DBLP dataset
# this can use either deduplicate function above, the NumPy approach may be faster a bit though
dblp_dupe, dblp_unique = deduplicate_np (dblp)

#save the cleaned dataset for phased processing
dblp_unique.to_csv(fname_dblp_clean, index=False)
dblp_dupe.to_csv(fname_dblp_dupe, index=False)

# load and prepare the Scholar dataset

scholar = load_prepare_dataset(fname_scholar)
scholar.rename(columns={'ROW_ID':'Row_ID'}, inplace = True)

# deduplicate the Scholar dataset (optional)
# the DataFrame approach would have memory issue due to the data size

#scholar_dupe, scholar_unique = deduplicate_np (scholar)
# To bypass the deduplication for Scholar dataset, uncomment the line below and comment the line above
scholar_unique = scholar
scholar_unique.to_csv(fname_scholar_clean, index=False)
#scholar_dupe.to_csv(fname_scholar_dupe, index=False)


# Resolve the linkage between the DBLP and Scholar datasets

#dblp_scholar_links = link_records (dblp_unique, scholar_unique)

# Write the result links to output file

#dblp_scholar_links.to_csv(fname_result)

# The end of the solution
