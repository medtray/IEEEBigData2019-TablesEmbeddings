import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import numpy as np
from utils import *
import json
from collections import Counter

from random import shuffle

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

wikitables_folder='wikitables'
mcon_data_folder='mcon_data'
output_folder=os.path.join(wikitables_folder,mcon_data_folder)
data_folder='/path/to/tables'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(output_folder+'/X'):
    os.mkdir(output_folder+'/X')

if not os.path.exists(output_folder+'/Y'):
    os.mkdir(output_folder+'/Y')

list_of_categories=os.listdir(data_folder)
nb_files=len(list_of_categories)

mylist = list(range(nb_files))
print("start shuffle")
shuffle(mylist)
print("shuffle done")

list_of_categories=np.array(list_of_categories)[mylist]
nb_files_in_training=len(list_of_categories)
list_of_categories=list_of_categories[:nb_files_in_training]
train_file_list = output_folder +'/list_of_categories'
np.save(train_file_list,list_of_categories)
skip_gram_data = []
size_threshold_of_list=10000000
batch_size = 1000000
total_size_of_data=0
nb_of_tables_processed=0
value_context=0
same_attribute_context=0
other_attributes_context=0
description_context=0
input_word2int = {}
input_int2word = {}
output_word2int = {}
output_int2word = {}

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def save_chunks(skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data):
    x_train = [w[0] for w in skip_gram_data]
    y_train = [w[1] for w in skip_gram_data]

    size_input_dict=len(input_word2int)
    index=-1

    for word in x_train:
        if word not in input_word2int:
            index+=1
            input_word2int[word]=size_input_dict+index
            input_int2word[size_input_dict+index]=word

    size_output_dict = len(output_word2int)
    index = -1
    for i, word in enumerate(y_train):
        if word not in output_word2int:
            index += 1
            output_word2int[word] = size_output_dict + index
            output_int2word[size_output_dict + index] = word

    mylist = list(range(len(x_train)))
    print("start shuffle")
    shuffle(mylist)
    print("shuffle done")

    x_train = [input_word2int[x_train[w]] for w in mylist]
    y_train = [output_word2int[y_train[w]] for w in mylist]
    total_size_of_data += len(skip_gram_data)
    del skip_gram_data
    skip_gram_data = []

    nb_batches = len(x_train) // batch_size
    rest_of_samples = len(x_train) % batch_size

    counter = 0

    inside_x = os.listdir(output_folder + '/X')
    start_index = len(inside_x)

    for cb in range(1, nb_batches + 1):
        x_train_batch = x_train[batch_size * (cb - 1):batch_size * cb]
        y_train_batch = y_train[batch_size * (cb - 1):batch_size * cb]

        counter += len(x_train_batch)

        file_x_train = output_folder + '/X/x_train_' + str(cb + start_index)
        file_y_train = output_folder + '/Y/y_train_' + str(cb + start_index)

        np.save(file_x_train, x_train_batch)
        del x_train_batch
        np.save(file_y_train, y_train_batch)
        del y_train_batch


    if rest_of_samples>0:
        file_x_train = output_folder + '/X/x_train_' + str(nb_batches + 1 + start_index)
        file_y_train = output_folder + '/Y/y_train_' + str(nb_batches + 1 + start_index)
        x_train_batch = x_train[batch_size * nb_batches:]
        counter += len(x_train_batch)

        np.save(file_x_train, x_train_batch)
        del x_train_batch
        y_train_batch = y_train[batch_size * nb_batches:]
        np.save(file_y_train, y_train_batch)
        del y_train_batch

    print("size of data in chunks", counter)
    del x_train
    del y_train

    return skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data



for jj,category in enumerate(list_of_categories):
    print(jj)
    path = os.path.join(data_folder, category)

    with open(path) as f:
        dt = json.load(f)

    for key in dt:
        nb_of_tables_processed+=1
        table=dt[key]
        attributes=table['title']
        pgTitle=table['pgTitle']
        secondTitle=table['secondTitle']
        caption=table['caption']
        data=table['data']
        data_csv = pd.DataFrame(data, columns=attributes)
        num_indices=table['numericColumns']
        num_attributes=np.array(attributes)[num_indices]

        desc_name = pgTitle + ' ' + secondTitle+' '+caption
        desc_name_tokens = preprocess(desc_name, 'description')

        desc_name_tokens = [w + '_d' for w in desc_name_tokens]

        attributes_to_use = []
        tokens_of_attributes = []
        values_to_use = []

        attributes = list(data_csv)

        inter_att = ' '.join(attributes)
        att_tokens_inter = preprocess(inter_att, 'attribute')

        if len(att_tokens_inter)==0:
            data_csv = data_csv.transpose()
            data_csv_array = np.array(data_csv)
            if data_csv_array.size>0:
                attributes = data_csv_array[0, :]
                data_csv = pd.DataFrame(data_csv_array, columns=attributes)

                data_csv = data_csv.drop([0], axis=0).reset_index(drop=True)
            else:
                data_csv = data_csv.transpose()


        attribute_counter=Counter(attributes)

        data_csv=data_csv.replace(r'^\s*$', 'missing value', regex=True)

        for att in attribute_counter:

            contain_numbers = False

            try:
                float(att)
                att_is_numerical = True
                break
            except ValueError:
                att_is_numerical = False

            if len(att)>0 and not att_is_numerical:
                if attribute_counter[att]==1:
                    df_2 = data_csv[att].copy()

                    df_2.fillna('missing value', inplace=True)
                    elements = df_2
                    elements = [e for e in elements if e != 'missing value']
                    unique_values_of_attribute = elements

                else:
                    df_2 = data_csv[att].copy()

                    df_2.fillna('missing value', inplace=True)
                    elements = df_2
                    all_elements = elements.values
                    all_elements = all_elements.reshape([-1])
                    all_elements = [e for e in all_elements if e != 'missing value']
                    unique_values_of_attribute=all_elements

                # tokenize values
                tok_val = []

                for val in unique_values_of_attribute:

                    val_inter = preprocess(val, 'value')


                    val_inter = [w + '_v' for w in val_inter]
                    tok_val += val_inter

                attributes_to_use.append(att)

                att_tokens = preprocess(att, 'attribute')

                tokens_of_attributes.append(att_tokens)

                values_to_use.append(tok_val)

        nb_of_attributes = len(tokens_of_attributes)

        for att, att_to_use in enumerate(tokens_of_attributes):
            for tok_att_index,tok_att in enumerate(att_to_use):
                for uvs_atts in values_to_use[att]:
                    skip_gram_data.append([tok_att, uvs_atts])
                    value_context+=1
                    if len(skip_gram_data) == size_threshold_of_list:
                        skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data=save_chunks(skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data)

                for tok_name in desc_name_tokens:
                    skip_gram_data.append([tok_att, tok_name])
                    description_context+=1
                    if len(skip_gram_data) == size_threshold_of_list:
                        skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data=save_chunks(skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data)

                tok_att_context=att_to_use.copy()
                del tok_att_context[tok_att_index]
                tok_att_context = [w + '_a' for w in tok_att_context]

                for tok_att_cont in tok_att_context:
                    skip_gram_data.append([tok_att, tok_att_cont])
                    same_attribute_context+=1
                    if len(skip_gram_data) == size_threshold_of_list:
                        skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data=save_chunks(skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data)

                tok_att_other_attributes_c=tokens_of_attributes.copy()
                del tok_att_other_attributes_c[att]

                for att_context in tok_att_other_attributes_c:
                    att_context = [w + '_o' for w in att_context]

                    for token_of_oac in  att_context:
                        skip_gram_data.append([tok_att, token_of_oac])
                        other_attributes_context+=1
                        if len(skip_gram_data) == size_threshold_of_list:
                            skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data = save_chunks(
                                skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data)




if len(skip_gram_data)>0:
    skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data = save_chunks(skip_gram_data, input_word2int,input_int2word,output_word2int,output_int2word, total_size_of_data)


del skip_gram_data


print("total number of tables processed ",nb_of_tables_processed)

print("size of training data is ",total_size_of_data)

print("value context ",value_context)
print("description context ",description_context)
print("same attribute context ",same_attribute_context)
print("other attributes context ",other_attributes_context)

print("size of input dictionary is ",len(input_word2int))

np.save(output_folder+'/input_word2int_with_context', input_word2int)
np.save(output_folder+'/input_int2word_with_context', input_int2word)

print("size of output dictionary is ",len(output_word2int))

np.save(output_folder+'/output_word2int_with_context', output_word2int)
np.save(output_folder+'/output_int2word_with_context', output_int2word)

text_file = open(output_folder+"/done.txt", "w")
text_file.write("done")
text_file.close()
print('done')



