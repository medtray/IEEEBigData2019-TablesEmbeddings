
import os

import json
import subprocess
from collections import Counter
import pandas as pd
from utils import *

wikitables_folder='wikitables'
ranking_folder=os.path.join(wikitables_folder,'ranking_results')
text_file=os.path.join(ranking_folder,'rank_mcon.txt')
text_file = open(text_file, "r")
lines = text_file.readlines()

list_lines=[]
for line in lines:
    #print(line)
    line=line[0:len(line)-1]
    aa=line.split('\t')
    list_lines.append(aa)

inter = np.array(list_lines)

best_ndcg=0

for w_pgTitle in np.arange(0.4, 0.44, 0.1):
    for w_secondTitle in np.arange(0., 0.11, 0.05):
        for w_caption in np.arange(0, 0.11, 0.05):
            for w_attributes in np.arange(0, 0.11, 0.05):
                for w_att_input in np.arange(0., 0.51, 0.05):
                    for w_attc in np.arange(0, 0.51, 0.05):
                        for w_descc in np.arange(0, 0.11, 0.05):
                            for w_valc in np.arange(0, 0.11, 0.05):
                                for w_data  in np.arange(0.0, 0.11, 0.05):

                                    coef=np.array([w_pgTitle,w_attributes,w_secondTitle,w_caption,w_att_input,w_attc,w_descc,w_valc,w_data])

                                    inter2=[]
                                    for item in inter:
                                        row=item[0]

                                        row=row.split(' ')
                                        distance=row[4]
                                        distance=distance.split('+')
                                        distance=[float(i) for i in distance]

                                        distance=np.array(distance)
                                        final_dist = np.matmul(distance, coef)

                                        row[4]=str(final_dist)
                                        inter2.append(row)

                                    inter2=np.array(inter2)

                                    file_name = os.path.join(ranking_folder,'scores_gs.txt')

                                    np.savetxt(file_name, inter2, fmt="%s")
                                    batcmd = "cd " + ranking_folder + " && ./trec_eval -m ndcg_cut.20 qrels.txt scores_gs.txt"
                                    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
                                    res = result.split('\t')
                                    ndcg = float(res[2])

                                    if ndcg > best_ndcg:
                                        best_ndcg = ndcg

                                        line = 'w_attributes ' + str(w_attributes) + ' w_pgTitle ' + str(w_pgTitle) + ' w_secondTitle ' + str(
                                            w_secondTitle) + ' w_caption ' + str(w_caption)  + ' w_att_input ' + str(
                                            w_att_input) + ' w_attc ' + str(w_attc) + ' w_descc ' + str(w_descc) + ' w_valc ' + str(
                                            w_valc) +' w_data ' + str(w_data)+' result ' + str(ndcg)

                                        print(line)
