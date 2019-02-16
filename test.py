import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime as DT
import collections
import argparse

from matplotlib.dates import date2num
import pickle
import pandas as pd
import csv
import warnings
from model_date_portal import HRAlgorithm
from heartspace import heartspaceNN
import random

warnings.filterwarnings('always')


def load_data(file_name, convert2array=True):
    fdata = open(file_name, 'rb')
    txt_file_name = [line.rstrip() for line in fdata]
    values = [];
    for idx,row in enumerate(txt_file_name):
        row = row.decode('cp1252').strip('\r\n').split(' ')
        values.append(row)
    fdata.close()
    if convert2array:
        return np.array(values, dtype='f')
    else:
        return values

def plot_motif_shape(data0, data1, data2, start_point1, start_point2, end_point1, end_point2):
    plt.figure()
    grid = plt.GridSpec(4, 4, wspace=0.1, hspace=0.3)
    # plt.subplot(4, 1, 1)
    plt.subplot(grid[0, 0:])
    plt.title("Origin-plot")
    plt.plot(data0, data1, '-')
    # plt.subplot(4, 1, 2)
    plt.subplot(grid[1, 0:])
    plt.title("Lowpass-filter")
    plt.plot(data0[50:], data2[50:], 'g-', linewidth=2, label='filtered data')
    # plt.subplot(4, 1, 3)
    plt.subplot(grid[2, 0:2])
    plt.title("Zoom-out")
    plt.plot(data0[start_point1:end_point1], data2[start_point1:end_point1], 'g-', linewidth=2,
             label='filtered data')
    plt.plot(data0[start_point1:end_point1], data1[start_point1:end_point1], '--',
             linewidth=0.5,
             label='filtered data')
    plt.subplot(grid[2, 2:4])
    plt.title("Zoom-out")
    plt.plot(data0[start_point2:end_point2], data2[start_point2:end_point2], 'g-', linewidth=2,
             label='filtered data')
    plt.plot(data0[start_point2:end_point2], data1[start_point2:end_point2], '--', linewidth=0.5,
             label='filtered data')
    plt.savefig('color_img3.jpg')

def plot_ae_motif(img1, img2, img3, fileName):
    plt.figure()
    grid = plt.GridSpec(1, 3, wspace=0.1, hspace=0.3)
    plt.subplot(grid[0, 0])
    plt.imshow(np.squeeze(img1,axis=-1))
    plt.title("Origin-fig")
    plt.subplot(grid[0, 1])
    plt.imshow(np.squeeze(img2,axis=-1))
    plt.title("resized-fig")
    plt.subplot(grid[0, 2])
    plt.imshow(np.squeeze(img3,axis=-1))
    plt.title("reconstructed-fig")
    plt.savefig(fileName)
    plt.close()

def date2weekday(dt):
    DayL = ['Mon', 'Tues', 'Wednes', 'Thurs', 'Fri', 'Satur', 'Sun']
    year, month, day = (int(x) for x in dt.split('-'))
    answer = DT.date(year, month, day).weekday()
    return DayL[answer]

def load_motif(file): #generate image
    data = pd.read_csv(file)
    print(data)
    id_set = data['id'].unique().tolist()
    print(id_set)
    idd_tv = data.groupby(by = ['id', 'date'])['time', 'rate'].apply(lambda x: x.values.tolist()).to_dict()
    mtx = {}
    for (id, d) in idd_tv:
        for v in idd_tv[(id, d)]:
            try:
                mtx[id][d][int(v[0][:2]), int(v[0][3:5])] = v[1]
            except:
                try:
                    mtx[id][d] = np.zeros([24, 60])
                    mtx[id][d][int(v[0][:2]), int(v[0][3:5])] = v[1]
                except:
                    mtx[id] = {}
                    mtx[id][d] = np.zeros([24, 60])
                    mtx[id][d][int(v[0][:2]), int(v[0][3:5])] = v[1]
    return mtx

def convert2idx(words, offset, is_set = False):
    dictionary = {}
    for word in words:
        dictionary[word] = len(dictionary) + offset
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def load_header_data(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)

        header = [h.lower() for h in header]
        columns = {h:[] for h in header}
        for row in csv_reader:
            for h, v in zip(header, row):
                if h == 'id' or h == 'date':
                    columns[h].append(v)
                else:
                    try:
                        columns[h].append(float(v))
                    except ValueError:
                        columns[h].append(v)
        return columns


def gen_batchs(mtx, central_id, pos_id, neg_id, batch_size):
    rng_state = np.random.get_state()
    np.random.shuffle(central_id)
    np.random.set_state(rng_state)
    np.random.shuffle(pos_id)
    np.random.set_state(rng_state)
    np.random.shuffle(neg_id)
    xs_batches = len(central_id) // batch_size
    for idx in range(xs_batches):
        idx_begin = idx * batch_size
        idx_end = (idx + 1) * batch_size
        batch_central_Image = []
        batch_pos_motifImage = []
        batch_neg_motifImage = []
        batch_central_date = []

        for _ in pos_id[idx_begin:idx_end]:
            tmp = []
            for(id, d) in _:
                tmp.append(mtx[id][d])
            batch_pos_motifImage.append(np.array(tmp))


        for _ in central_id[idx_begin:idx_end]:
            tmp = []; tmp_d = []
            for(id, d) in _:
                tmp.append(mtx[id][d]); tmp_d.append(dic_dv[d])
            batch_central_Image.append(np.array(tmp))
            tmp_max_d = max(tmp_d)
            batch_central_date.append([tmp_max_d-_ for _ in tmp_d])

        for _ in neg_id[idx_begin:idx_end]:
            tmp = []
            for(id, d) in _:
                tmp.append(mtx[id][d])
            batch_neg_motifImage.append(np.array(tmp))
        yield (np.array(batch_central_Image), np.array(batch_central_date), np.array(batch_pos_motifImage), np.array(batch_neg_motifImage))

def gen_epochs(n, mtx, num_input, num_pos, num_neg, batch_size):
    central_id = []
    pos_id = []
    neg_id = []
    for id in mtx:
        ds = list(mtx[id].keys())
        for sample_idx in range(1000):

            sampled_keys = sample(ds, num_pos + num_input)
            pos_id.append([(id, d) for d in sampled_keys[:num_pos]])
            central_id.append([(id, d) for d in sampled_keys[num_pos:]])
            sampled_keys = []
            for _ in range(3 * num_neg):
                random_id = sample(list(mtx.keys()), 1)[0]
                random_date = sample(list(mtx[random_id].keys()), 1)[0]
                sampled_keys.append((random_id, random_date))
                if len(sampled_keys) == num_neg:
                    neg_id.append(sampled_keys)
                    break
            neg_id.append(sampled_keys)
    for i in range(n):
        yield (gen_batchs(mtx, central_id, pos_id, neg_id, batch_size))

def alys_histogram(image):
    _ = np.int32(np.array(image).reshape(-1)/50); len_ = len(_)
    dict_ = collections.Counter(_)
    dict_= {k:round(dict_[k]/len_,2) for k in dict_}
    return dict_
'###############################'
DateFormat = lambda v: DT.datetime.fromordinal(int(v)).strftime("%Y-%m-%d")
Date2v = lambda date: int(date2num(DT.datetime.strptime(date.split(' ')[0], "%Y-%m-%d").replace(minute=0).replace(second=0)))
sample = lambda x,k: np.array(x)[np.random.randint(0,len(x),size=(k))]
'###############################'
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, default='heartspace' ,help=u"model name")
args = parser.parse_args()
'******************************************************************'
with open('./sample_hr.pkl', 'rb') as f:
    [id_set, mtx] = pickle.load(f)

dates = list(set([d for id in mtx for d in mtx[id]]))
dic_dv = {d:Date2v(d) for d in dates}

dict_user, reverse_dict_user = convert2idx(id_set, 1, is_set=False)
dict_user['nan'] = 0; reverse_dict_user[0] = 'nan'
'*******************Set model************************************'
model_dir = './tmp_data/model/'+args.m
batch_size = 64; num_input = 6; num_pos = 2; num_neg = 4; n_user = len(id_set)

alg = HRAlgorithm(num_pos, num_neg, featureDimension=[24, 60], penalty_rate=0,
                  v_scope='', keep_rate=0.9, obv=50, batch_size=batch_size, learning_rate=5e-4,
                  attention_size=32, n_filters=[32, 64, 64, 128, 128], filter_sizes=[9, 7, 7, 5, 5],
                  image_featuredim=64, n_mlp=1, NN=heartspaceNN, num_input=num_input,
                  overall_indicator=0, n_class={})
'*******************train************************************'
loss_list = []
avg_loss = 0
alg.create(pretrain_flag=0, save_file=model_dir)
for epoch_idx, epoch in enumerate(gen_epochs(10, mtx, num_input, num_pos, num_neg, batch_size)):
    for batch_central_motifImage, batch_central_date, batch_pos_motifImage, batch_neg_motifImage in epoch:
        step, loss_ = alg.feedbatch(batch_central_motifImage, batch_central_date, batch_pos_motifImage, batch_neg_motifImage)
        avg_loss += loss_
        if step % 50 == 0:
            avg_loss /= 50
            loss_list.append(np.round(avg_loss, 3))
            print('(epoch %s, step %s): avg_loss = %.4f ' % (epoch_idx, step, avg_loss))
            avg_loss = 0
    print('')

    alg.save_weight(model_dir)
print('loss_list', loss_list)
'*******************output embeds************************************'
avg_out_file = './agg_' + str(args.m) + '_embed.txt'
daily_out_file = './daily_' + str(args.m) + '_embed.txt'

fout_avg = open(avg_out_file, 'w')
fout_daily = open(daily_out_file, 'w')
for id in mtx:
    dates = [];
    date_vs = [];
    inputs = []
    for k, v in mtx[id].items():
        dates.append(k)
        date_vs.append(dic_dv[k])
        inputs.append(v)
    max_date = max(date_vs)
    date_vs = [max_date - _ for _ in date_vs]

    agg_embed, _ = alg.getOverallEmbed([inputs], [date_vs])
    agg_embed = agg_embed[0]
    daily_embed = alg.getDailyEmbed(inputs)
    print('agg_embed', agg_embed[:5])
    row = id + ':' + (',').join([str(round(_, 3)) for _ in agg_embed]) + '\n'
    fout_avg.write(row)
    for date, vec in zip(dates, daily_embed):
        row = id + ' ' + str(date) + ':' + (',').join([str(round(_, 3)) for _ in vec]) + '\n'
        fout_daily.write(row)
fout_avg.close()
fout_daily.close()