import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText as ft
import os
import json
from collections import Counter
import pandas as pd
from utils import *
config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

print('begin fit model')
model=ft.load_fasttext_format("wiki.en.bin")
print('finish fit model')

wikitables_folder='wikitables'
file_to_index='mcon_predictions.json'
path=os.path.join(wikitables_folder,file_to_index)

with open(path) as f:
    dt = json.load(f)

data_csv=pd.read_csv(os.path.join(wikitables_folder,'features2.csv'))
attributes = list(data_csv)
test_data=data_csv['table_id']
query=data_csv['query']
relevance=data_csv['rel']
query_ids=list(set(data_csv['query_id']))
text_file = open("multi_field.txt", "r")
lines = text_file.readlines()
mcon_data_folder='mcon_data'
chunks_folder=os.path.join(wikitables_folder,mcon_data_folder)

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


input_word2int=np.load(chunks_folder+'/input_word2int_with_context.npy',allow_pickle=True)
input_word2int=input_word2int[()]
input_vocab_size = len(input_word2int)
input_int2word=np.load(chunks_folder+'/input_int2word_with_context.npy',allow_pickle=True)
input_int2word=input_int2word[()]
output_int2word=np.load(chunks_folder+'/output_int2word_with_context.npy',allow_pickle=True)
output_int2word=output_int2word[()]
output_vocab_size = len(output_int2word)
model_dir = chunks_folder + '/skip_gram_model_with_context'
EMBEDDING_DIM = 100
num_sampled = 10000
checkpoint_name = 'model.ckpt'
checkpoint_path = os.path.join(model_dir, checkpoint_name)
top_predictions=50
additional_att=50
top_tokens=3

graph = tf.Graph()

with graph.as_default():
    x = tf.compat.v1.placeholder(tf.int32, shape=[None])
    x_onehot = tf.one_hot(x, input_vocab_size)

    W1 = tf.Variable(tf.compat.v1.random_normal([input_vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.compat.v1.random_normal([EMBEDDING_DIM]))  # bias
    hidden_representation = tf.add(tf.matmul(x_onehot, W1), b1)

    W2 = tf.Variable(tf.compat.v1.random_normal([output_vocab_size, EMBEDDING_DIM]))
    b2 = tf.Variable(tf.compat.v1.random_normal([output_vocab_size]))

    saver = tf.compat.v1.train.Saver(max_to_keep=2)


with tf.compat.v1.Session(graph=graph,config=config) as sess:
    with tf.device('/gpu:0'):
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        bb = [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]
        w1 = bb[0]
        w1 = sess.run(w1)
        word2vect = w1
        op_to_restore = graph.get_tensor_by_name("MatMul:0")

        input = graph.get_tensor_by_name("Placeholder:0")
        w2 = graph.get_tensor_by_name('Variable_2:0')

        prediction = tf.nn.softmax(tf.add(tf.matmul(op_to_restore, tf.transpose(bb[2])), bb[3]))
        top_k_values, top_k_indices = tf.nn.top_k(prediction, k=top_predictions)

    def late_fusion_ranking_final():

        final_table = []

        for q in qq:
            print(q)

            indexes = [i for i, x in enumerate(queries_id) if x == q]
            indices = data_csv[data_csv['query_id'] == q].index.tolist()

            inter = np.array(list_lines)[indexes]
            inter2 = []
            test_query = list(query[indices])[0]
            query_tokens = preprocess(test_query, 'description')

            for item in inter:
                if item[2] in test_data:

                    table = dt[item[2]]
                    pgTitle_feat = table['pgTitle']
                    if len(pgTitle_feat)>0:
                        pgTitle_feat=pgTitle_feat.split(' ')
                    else:
                        pgTitle_feat=[]


                    secondTitle_feat = table['secondTitle']
                    if len(secondTitle_feat)>0:
                        secondTitle_feat=secondTitle_feat.split(' ')
                    else:
                        secondTitle_feat=[]

                    caption_feat = table['caption']
                    if len(caption_feat)>0:
                        caption_feat=caption_feat.split(' ')
                    else:
                        caption_feat=[]

                    data_feat = table['data']
                    if len(data_feat) > 0:
                        data_feat = data_feat.split(' ')
                    else:
                        data_feat = []

                    values_context=table['values+']
                    values_context=values_context.split(' ')

                    description_context=table['description+']
                    description_context = description_context.split(' ')

                    att_context=table['attributes+']
                    att_context=att_context.split(' ')

                    closest=table['attributes++']
                    closest=closest.split(' ')

                    original_attributes = table['attributes']
                    original_attributes = original_attributes.split(' ')
                    original_attributes=[tok for tok in original_attributes if tok in input_word2int]

                    word2vec_att = [word2vect[input_word2int[tok]] for tok in original_attributes]

                    if len(word2vec_att) == 1:

                        element1 = np.array(word2vec_att).reshape(1, -1)

                    else:
                        element1 = np.array(word2vec_att)

                    word2vec_query = [word2vect[input_word2int[tok]] for tok in query_tokens if tok in input_word2int]

                    if len(word2vec_query) == 1:

                        element2 = np.array(word2vec_query).reshape(1, -1)

                    else:
                        element2 = np.array(word2vec_query)

                    cos_lib = cosine_similarity(element2, element1)

                    #cos_lib=np.random.rand(len(query_tokens), len(original_attributes))
                    ids = cos_lib.argsort()

                    att_indices=[]

                    for row in ids:
                        att_indices += list(row)[::-1][0:top_tokens]

                    att_indices=list(set(att_indices))

                    all_att_tokens = np.take(original_attributes, att_indices)
                    x_train_batch = [input_word2int[i] for i in all_att_tokens]

                    x_train_batch=np.asarray(x_train_batch)

                    values_context = []
                    description_context = []
                    att_context = []

                    closest = []

                    if len(x_train_batch) > 0:
                        word2vec_att = [word2vect[int_tok] for int_tok in x_train_batch]

                        if len(word2vec_att) == 1:

                            element1 = np.array(word2vec_att).reshape(1, -1)

                        else:
                            element1 = np.array(word2vec_att)

                        cos_lib = cosine_similarity(element1, word2vect)
                        ids = cos_lib.argsort()

                        for row in ids:
                            aa = row[::-1][1:additional_att]
                            closest += [input_int2word[i] for i in aa]

                        feed_dict = {input: x_train_batch}
                        batch_result = sess.run(top_k_indices, feed_dict)

                        unique_context = []

                        for sample in batch_result:
                            sample_context = [output_int2word[i] for i in sample]

                            unique_context = list(set(unique_context + sample_context))


                        for word in unique_context:
                            if word[-1] == 'd':
                                word = word[0:len(word) - 2]
                                description_context.append(word)

                            elif word[-1] == 'v':
                                word = word[0:len(word) - 2]
                                values_context.append(word)

                            else:
                                word = word[0:len(word) - 2]
                                att_context.append(word)



                    att_feat = table['attributes']
                    att_feat = att_feat.split(' ')

                    def late_fusion_score(query_tokens,other_tokens):
                        word2vec_att = [word2vect[input_word2int[tok]] for tok in other_tokens if
                                        tok in input_word2int]

                        if len(word2vec_att) == 1:

                            element1 = np.array(word2vec_att).reshape(1, -1)

                        else:
                            element1 = np.array(word2vec_att)

                        word2vec_query = [word2vect[input_word2int[tok]] for tok in query_tokens if
                                          tok in input_word2int]

                        if len(word2vec_query) == 1:

                            element2 = np.array(word2vec_query).reshape(1, -1)

                        else:
                            element2 = np.array(word2vec_query)

                        if len(element1)>0 and len(element2)>0:

                            cos_lib = cosine_similarity(element2, element1)
                            #result=np.mean(cos_lib)
                            result=np.sum(np.amax(cos_lib,axis=1))
                            #result = np.sum(np.amax(cos_lib, axis=0))

                        else:
                            result=0
                        return result

                    def late_fusion_score_fasttext(query_tokens, other_tokens):
                        word2vec_att = []
                        for tok in other_tokens:
                            try:
                                word2vec_att.append(model.wv[tok])
                            except:
                                pass

                        if len(word2vec_att) == 1:

                            element1 = np.array(word2vec_att).reshape(1, -1)

                        else:
                            element1 = np.array(word2vec_att)

                        word2vec_query = [model.wv[tok] for tok in query_tokens]

                        if len(word2vec_query) == 1:

                            element2 = np.array(word2vec_query).reshape(1, -1)

                        else:
                            element2 = np.array(word2vec_query)

                        if len(element1) > 0 and len(element2) > 0:

                            cos_lib = cosine_similarity(element2, element1)
                            #result = np.mean(cos_lib)
                            result = np.sum(np.amax(cos_lib, axis=1))
                            #result = np.sum(np.amax(cos_lib, axis=0))

                        else:
                            result = 0
                        return result


                    distance = []

                    if len(pgTitle_feat) >0:

                        desc_score = late_fusion_score_fasttext(query_tokens, pgTitle_feat)
                        distance.append(desc_score)
                    else:
                        desc_score = 0
                        distance.append(desc_score)

                    if len(original_attributes) > 0:

                        distance.append(late_fusion_score(query_tokens, original_attributes))

                    else:
                        distance.append(desc_score)

                    if len(secondTitle_feat) > 0:

                        distance.append(late_fusion_score_fasttext(query_tokens, secondTitle_feat))
                    else:
                        distance.append(desc_score)

                    if len(caption_feat) > 0:

                        distance.append(late_fusion_score_fasttext(query_tokens, caption_feat))

                    else:
                        distance.append(desc_score)


                    if len(closest) > 0:
                        distance.append(late_fusion_score(query_tokens, closest))

                    else:
                        distance.append(desc_score)

                    if len(att_context) > 0:
                        distance.append(late_fusion_score(query_tokens, att_context))

                    else:
                        distance.append(desc_score)


                    if len(description_context) > 0:

                        distance.append(late_fusion_score_fasttext(query_tokens, description_context))

                    else:
                        distance.append(desc_score)

                    if len(values_context) > 0:

                        distance.append(late_fusion_score_fasttext(query_tokens, values_context))

                    else:
                        distance.append(desc_score)

                    if len(data_feat) > 0:

                        distance.append(late_fusion_score_fasttext(query_tokens, data_feat))

                    else:
                        distance.append(desc_score)

                    distance=[str(a) for a in distance]
                    distance='+'.join(distance)

                    item_inter=[i for i in item]
                    item_inter[4]=distance

                    inter2.append(item_inter)

            inter3 = np.array(inter2)

            for i in range(len(inter3)):
                inter3[i, 3] = 0

            final_table.append(inter3)

        final_table = np.concatenate(final_table, axis=0)
        print(len(final_table))

        ranking_folder = os.path.join(wikitables_folder, 'ranking_results')
        file_name = os.path.join(ranking_folder, 'rank_mcon.txt')
        np.savetxt(file_name, final_table, fmt="%s")
    late_fusion_ranking_final()