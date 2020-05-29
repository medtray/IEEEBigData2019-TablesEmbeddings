import os
import sys
sys.path.append('/home/mohamed/PycharmProjects/nordlys')
from nordlys.core.retrieval.elastic import Elastic
from nordlys.core.retrieval.elastic_cache import ElasticCache
from nordlys.config import PLOGGER
from nordlys.core.retrieval.scorer import *
import numpy as np
import json
import pandas as pd
from utils import *
import subprocess

index_name = "mcon_index"
es = ElasticCache(index_name)
wikitables_folder='wikitables'
data_csv=pd.read_csv(os.path.join(wikitables_folder,'features2.csv'))
ranking_folder=os.path.join(wikitables_folder,'ranking_results')
test_file=os.path.join(ranking_folder,'multi_field.txt')

attributes = list(data_csv)

test_data=data_csv['table_id']
query=data_csv['query']
relevance=data_csv['rel']

query_ids=list(set(data_csv['query_id']))

text_file = open(test_file, "r")
lines = text_file.readlines()

queries_id=[]
list_lines=[]
for line in lines:
    line=line[0:len(line)-1]
    aa=line.split('\t')
    queries_id+=[aa[0]]
    list_lines.append(aa)

queries_id = [int(i) for i in queries_id]
qq=np.sort(list(set(queries_id)))
test_data=list(test_data)


def generate_ranking(feature_used):
    params = {"fields": feature_used,
              # "fields": {"title": 0.2, "content": 0.8},
              # "__fields": ["content", "title"]
              }

    final_table = []

    print(feature_used)

    for q in qq:
        print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]

        query_tokens = preprocess(test_query, 'description')

        query_tokens = ' '.join(query_tokens)

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]

                file_name = 're_tables-' + file_number + '.json'

                table_name = 'table-' + file_number + '-' + table_number

                score = ScorerLM(es, query_tokens, params).score_doc(table_name)

                try:
                    float(score)
                except:
                    pass

                item[4] = score
                inter2.append(item)
                to_sort.append(score)

        order = np.argsort(to_sort)[::-1]

        inter3 = np.array(inter2)[order]

        for i in range(len(inter3)):
            inter3[i, 3] = i + 1

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    print(len(final_table))

    file_name = os.path.join(ranking_folder,'rank_' + feature_used + '.txt')

    np.savetxt(file_name, final_table, fmt="%s")




def generate_ranking_MLM(w_attributes,w_pgTitle,w_secondTitle,w_caption,w_data):



    total_ndcg=w_attributes+w_pgTitle+w_secondTitle+w_caption+w_data

    w_attributes=w_attributes/total_ndcg
    w_pgTitle=w_pgTitle/total_ndcg
    w_secondTitle=w_secondTitle/total_ndcg
    w_caption=w_caption/total_ndcg
    w_data=w_data/total_ndcg

    params = {
               "fields": {"attributes":w_attributes, "pgTitle":w_pgTitle,"secondTitle":w_secondTitle,"caption":w_caption,"data":w_data},
              }

    final_table = []

    for q in qq:
        #print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]

        query_tokens = preprocess(test_query, 'description')

        query_tokens=' '.join(query_tokens)

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]
                table_name = 'table-' + file_number + '-' + table_number
                score = ScorerMLM(es, query_tokens, params).score_doc(table_name)

                try:
                    float(score)
                except:
                    pass

                item[4] = score
                inter2.append(item)
                to_sort.append(score)

        order = np.argsort(to_sort)[::-1]

        inter3 = np.array(inter2)[order]

        for i in range(len(inter3)):
            inter3[i, 3] = i + 1

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    file_name = os.path.join(ranking_folder, 'rank_' + 'all' + '.txt')

    np.savetxt(file_name, final_table, fmt="%s")


def generate_ranking_MLM_AF(w_attributes,w_pgTitle,w_secondTitle,w_caption,w_data,w_att_input,w_attc,w_descc,w_valc):

    total_ndcg=w_attributes+w_pgTitle+w_secondTitle+w_caption+w_data+w_att_input+w_attc+w_descc+w_valc

    w_attributes=w_attributes/total_ndcg
    w_pgTitle=w_pgTitle/total_ndcg
    w_secondTitle=w_secondTitle/total_ndcg
    w_caption=w_caption/total_ndcg
    w_data = w_data / total_ndcg
    w_att_input=w_att_input/total_ndcg
    w_attc = w_attc / total_ndcg
    w_descc = w_descc / total_ndcg
    w_valc = w_valc / total_ndcg

    params = {
               "fields": {"attributes":w_attributes, "pgTitle":w_pgTitle,"secondTitle":w_secondTitle,"caption":w_caption,"attributes++":w_att_input,"attributes+":w_attc,"description+":w_descc,"values+":w_valc,"data":w_data},
              }

    final_table = []

    for q in qq:
        #print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]

        query_tokens = preprocess(test_query, 'description')

        query_tokens=' '.join(query_tokens)

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]
                table_name = 'table-' + file_number + '-' + table_number
                score = ScorerMLM(es, query_tokens, params).score_doc(table_name)

                try:
                    float(score)
                except:
                    pass

                item[4] = score
                inter2.append(item)
                to_sort.append(score)

        order = np.argsort(to_sort)[::-1]

        inter3 = np.array(inter2)[order]

        for i in range(len(inter3)):
            inter3[i, 3] = i + 1

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    file_name = os.path.join(ranking_folder, 'rank_all_af.txt')

    np.savetxt(file_name, final_table, fmt="%s")


def generate_ranking_MLM_C(w_attributes,w_pgTitle,w_secondTitle,w_caption,w_data,w_context):

    total_ndcg=w_attributes+w_pgTitle+w_secondTitle+w_caption+w_data+w_context

    w_attributes=w_attributes/total_ndcg
    w_pgTitle=w_pgTitle/total_ndcg
    w_secondTitle=w_secondTitle/total_ndcg
    w_caption=w_caption/total_ndcg
    w_data=w_data/total_ndcg
    w_context=w_context/total_ndcg

    params = {
               "fields": {"attributes":w_attributes, "pgTitle":w_pgTitle,"secondTitle":w_secondTitle,"caption":w_caption,"data":w_data,"context":w_context},
              }

    final_table = []

    for q in qq:
        #print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]

        query_tokens = preprocess(test_query, 'description')

        query_tokens=' '.join(query_tokens)

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]

                file_name = 're_tables-' + file_number + '.json'

                table_name = 'table-' + file_number + '-' + table_number

                #print(table_name)

                #feat=test_table[feature_used]

                score = ScorerMLM(es, query_tokens, params).score_doc(table_name)

                try:
                    float(score)
                except:
                    pass

                item[4] = score
                inter2.append(item)
                to_sort.append(score)

        order = np.argsort(to_sort)[::-1]

        inter3 = np.array(inter2)[order]

        for i in range(len(inter3)):
            inter3[i, 3] = i + 1

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    file_name = os.path.join(ranking_folder, 'rank_all.txt')
    np.savetxt(file_name, final_table, fmt="%s")



def generate_ranking_BM25(feature_used):

    final_table = []

    print(feature_used)

    for q in qq:
        print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]

        result = es.search(test_query, feature_used)

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]

                table_name = 'table-' + file_number + '-' + table_number

                if table_name in result:
                    score=result[table_name]['score']
                else:
                    score=0

                item[4] = score
                inter2.append(item)
                to_sort.append(score)

        order = np.argsort(to_sort)[::-1]

        inter3 = np.array(inter2)[order]

        for i in range(len(inter3)):
            inter3[i, 3] = i + 1

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    file_name = os.path.join(ranking_folder, 'rank_bm25_'+ feature_used + '.txt')
    np.savetxt(file_name, final_table, fmt="%s")


def generate_ranking_BM25_all():

    final_table = []

    for q in qq:
        print(q)
        indexes = [i for i, x in enumerate(queries_id) if x == q]
        indices = data_csv[data_csv['query_id'] == q].index.tolist()

        inter = np.array(list_lines)[indexes]
        inter2 = []
        to_sort = []

        test_query = list(query[indices])[0]


        result_att = es.search(test_query, 'attributes')
        result_pgTitle = es.search(test_query, 'pgTitle')
        result_secondTitle = es.search(test_query, 'secondTitle')
        result_caption = es.search(test_query, 'caption')
        result_data = es.search(test_query, 'data')

        result_att_input = es.search(test_query, 'attributes++')
        result_attc = es.search(test_query, 'attributes+')
        result_desc = es.search(test_query, 'description+')
        result_valc = es.search(test_query, 'values+')

        for item in inter:
            if item[2] in test_data:
                table=item[2]
                inter = table.split("-")
                file_number = inter[1]
                table_number = inter[2]

                table_name = 'table-' + file_number + '-' + table_number

                distance = []

                if table_name in result_pgTitle:
                    distance.append(result_pgTitle[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_att:
                    distance.append(result_att[table_name]['score'])
                else:
                    distance.append(0)



                if table_name in result_secondTitle:
                    distance.append(result_secondTitle[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_caption:
                    distance.append(result_caption[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_att_input:
                    distance.append(result_att_input[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_attc:
                    distance.append(result_attc[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_desc:
                    distance.append(result_desc[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_valc:
                    distance.append(result_valc[table_name]['score'])
                else:
                    distance.append(0)

                if table_name in result_data:
                    distance.append(result_data[table_name]['score'])
                else:
                    distance.append(0)

                distance = [str(a) for a in distance]
                distance = '+'.join(distance)

                item_inter = [i for i in item]
                item_inter[4] = distance

                inter2.append(item_inter)

        inter3 = np.array(inter2)

        for i in range(len(inter3)):
            inter3[i, 3] = 0

        final_table.append(inter3)

    final_table = np.concatenate(final_table, axis=0)
    print(len(final_table))

    file_name = os.path.join(ranking_folder, 'rank_bm25.txt')
    np.savetxt(file_name, final_table, fmt="%s")

#parameters should be found using grid search on training data
w_attributes=0.2
w_pgTitle=0.3
w_secondTitle=0.1
w_caption=0.2
w_data=0.1
w_att_input=0.1
w_attc=0.1
w_descc=0.1
w_valc=0.1

#generate_ranking_MLM_AF(w_attributes,w_pgTitle,w_secondTitle,w_caption,w_data,w_att_input,w_attc,w_descc,w_valc)
generate_ranking_BM25_all()


# batcmd="cd "+ranking_folder+" && ./trec_eval -m ndcg_cut.20 qrels.txt rank_bm25.txt"
# result = subprocess.check_output(batcmd, shell=True,encoding='cp437')
# res=result.split('\t')
# ndcg=float(res[2])
# print(ndcg)