# ds_assignment
Data Scientist Candidate Assignment

Created on Fri Mar  9 09:50:56 2018

@author: Guofeng.Zhao

This assignment involve Deduplication and Record Linkage problems. Since there 
is no labelled data to train a ML model, I opted to implement this task using 
the very basic tools available in Python, NumPy and Pandas. This is considered
a pretty "traditional" approach.

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

The script "src/resolution.py" is the all-in-one resolution, which starts with 
helper functipons, then followed by the processing flow which starts from 
the comment line " ===== MAIN FLOW STARTS FROM HERE =====".

To run this script, go to the src directory and run "python resolution.py".
It may take about 3 hours on a moderate power laptop PC. 
A pre-run result is saved at 
"outputs/DBLP_Scholar_perfectMapping_GuofengZhao.csv"

There are also two segmented scripts available which allow to run thhe 
solution in two phases -
* src/phase1_deduplicate.py: Cleanse and deduplicate the datasets and persist to 
                         intermediate csv files
*        src/phase2_link.py: Resolve the record linkage and generate the final
                         result

