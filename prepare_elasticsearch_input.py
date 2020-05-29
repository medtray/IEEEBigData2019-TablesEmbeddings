from random import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import Counter
from utils import *
from tqdm import tqdm
config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

wikitables_folder='wikitables'
mcon_data_folder='mcon_data'
chunks_folder=os.path.join(wikitables_folder,mcon_data_folder)
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
top_predictions=20
additional_att=20
data_folder='/home/mohamed/PycharmProjects/Data-Search-Project/tables_redi2_1'
data_csv=pd.read_csv(os.path.join(wikitables_folder,'features2.csv'))
test_data=data_csv['table_id']
checkpoint_name = 'model.ckpt'
checkpoint_path = os.path.join(model_dir, checkpoint_name)
list_of_categories=os.listdir(data_folder)
nb_files=len(list_of_categories)
mylist = list(range(nb_files))
print("start shuffle")
shuffle(mylist)
print("shuffle done")
list_of_categories=np.array(list_of_categories)[mylist]
nb_files_in_training=2
list_of_categories=list_of_categories[:nb_files_in_training]

docs={}
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


    def prepare_table(test_table):

        attributes = test_table['title']
        pgTitle = test_table['pgTitle']
        secondTitle = test_table['secondTitle']
        caption = test_table['caption']
        data = test_table['data']

        pgTitle = preprocess(pgTitle, 'description')
        pgTitle = ' '.join(pgTitle)
        secondTitle = preprocess(secondTitle, 'description')
        secondTitle = ' '.join(secondTitle)
        caption = preprocess(caption, 'description')
        caption = ' '.join(caption)

        data_csv = pd.DataFrame(data, columns=attributes)

        attributes = list(data_csv)

        inter_att = ' '.join(attributes)
        att_tokens_inter = preprocess(inter_att, 'attribute')

        if len(att_tokens_inter) == 0:
            data_csv = data_csv.transpose()
            # vec_att = np.array(attributes).reshape(-1, 1)
            data_csv_array = np.array(data_csv)
            # data_csv_array = np.concatenate([vec_att, data_csv_array], axis=1)
            if data_csv_array.size > 0:
                attributes = data_csv_array[0, :]
                data_csv = pd.DataFrame(data_csv_array, columns=attributes)

                data_csv = data_csv.drop([0], axis=0).reset_index(drop=True)
            else:
                data_csv = data_csv.transpose()

        attribute_counter = Counter(attributes)

        all_att_tokens = []

        for att in attribute_counter:

            if len(att) > 0:
                att_tokens = preprocess(att, 'attribute')

                all_att_tokens = list(set(all_att_tokens + att_tokens))

        x_train_batch = np.asarray([input_word2int[tok] for tok in all_att_tokens if tok in input_word2int])

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

        all_att_tokens = []
        for att in attributes:
            att_tokens = preprocess(att, 'attribute')
            all_att_tokens += att_tokens

        attributes = ' '.join(all_att_tokens)

        data = data_csv.values

        closest = ' '.join(closest)
        description_context = ' '.join(description_context)
        values_context = ' '.join(values_context)
        att_context = ' '.join(att_context)

        data = ' '.join(y for x in data for y in x)

        return attributes,pgTitle,secondTitle,caption,data,closest,att_context,description_context,values_context

    with tqdm(total=len(list_of_categories)) as pbar0:
        for jj,category in enumerate(list_of_categories):
            path = os.path.join(data_folder, category)

            with open(path) as f:
                dt = json.load(f)

            for table_name in dt:

                test_table = dt[table_name]

                attributes, pgTitle, secondTitle, caption, data, closest, att_context, description_context, values_context=prepare_table(test_table)

                if table_name not in docs:
                    docs[table_name] = {}
                    docs[table_name]['attributes'] = attributes
                    docs[table_name]['pgTitle'] = pgTitle
                    docs[table_name]['secondTitle'] = secondTitle
                    docs[table_name]['caption'] = caption
                    docs[table_name]['data'] = data
                    docs[table_name]['attributes++'] = closest
                    docs[table_name]['attributes+'] = att_context
                    docs[table_name]['description+'] = description_context
                    docs[table_name]['values+'] = values_context
            pbar0.update(1)


    with tqdm(total=len(test_data)) as pbar:
        for jj, table in enumerate(test_data):

            inter = table.split("-")
            file_number = inter[1]
            table_number = inter[2]

            file_name = 're_tables-' + file_number + '.json'

            table_name = 'table-' + file_number + '-' + table_number

            path = os.path.join(data_folder, file_name)

            with open(path) as f:
                dt = json.load(f)

            test_table = dt[table_name]

            attributes, pgTitle, secondTitle, caption, data, closest, att_context, description_context, values_context = prepare_table(
                test_table)

            if table_name not in docs:
                docs[table_name] = {}
                docs[table_name]['attributes'] = attributes
                docs[table_name]['pgTitle'] = pgTitle
                docs[table_name]['secondTitle'] = secondTitle
                docs[table_name]['caption'] = caption
                docs[table_name]['data'] = data
                docs[table_name]['attributes++'] = closest
                docs[table_name]['attributes+'] = att_context
                docs[table_name]['description+'] = description_context
                docs[table_name]['values+'] = values_context

            pbar.update(1)

with open('mcon_predictions.json', 'w') as outfile:
    json.dump(docs, outfile)
