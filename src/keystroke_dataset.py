import pandas as pd
import numpy as np

from collections import defaultdict

from scipy.linalg import sqrtm
from scipy.spatial.distance import cityblock, mahalanobis, euclidean
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


subject = 'subject'
array = 'array'
target='target'


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

# Dwell time (the duration of time a key is pressed) and
# Flight time (the duration of time between "key up" and the next "key down")

fp='/home/dpetrovskyi/PycharmProjects/canstar/DSL-StrongPasswordData.csv'
password = '.tie5Roanl'

columns = ['H.period', 'DD.period.t',
           'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 'UD.i.e',
           'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r',
           'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o',
           'H.o', 'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n',
           'DD.n.l', 'UD.n.l', 'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return']


dwell_cols = ['H.period',
             'H.t',
             'H.i',
             'H.e',
             'H.five',
             'H.Shift.r',
             'H.o',
             'H.a',
             'H.n',
             'H.l',
             'H.Return']

flight_cols = ['UD.period.t',
               'UD.t.i',
               'UD.i.e',
               'UD.e.five',
               'UD.five.Shift.r',
               'UD.Shift.r.o',
               'UD.o.a',
               'UD.a.n',
               'UD.n.l',
               'UD.l.Return']

df = pd.read_csv(fp)

def decorrelation_matrix(m):
    covv = np.cov(m, rowvar=True)
    hol = sqrtm(covv)
    # hol = hol.real
    hol = np.linalg.inv(hol)
    # hol = np.transpose(hol)

    return hol

def manhaten_distance_after_decorelation(x,y, hol):
    x = np.dot(hol, x)
    y = np.dot(hol, y)

    return cityblock(x,y)

def add_array_col_to_df(df):
    cols = dwell_cols+flight_cols
    df['array'] = df.apply(lambda s: list(s[cols]), axis=1)
    for col in df.columns:
        if col not in ['array', 'subject']:
            del df[col]


def df_to_dict(df):
    res = defaultdict(list)
    df.apply(lambda s: res[s['subject']].append(s['array']), axis=1)

    return res


def calculate_manhaten_distance_after_decorelation_for_triple(a,b,c):
    m = np.array([a,b]).reshape(len(a),2)
    hol = decorrelation_matrix(m)

    return manhaten_distance_after_decorelation(a,c, hol), manhaten_distance_after_decorelation(b,c,hol)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_covariance_matrix_from_two_observations(a,b):
    m = np.array([a,b]).reshape(len(a),2)
    return np.cov(m, rowvar=True)


def create_balansed_data_set(df):
    add_array_col_to_df(df)
    m = {'a':[], 'b':[], 'target':[], 'cov':[], 'for_scaling_a':[], 'for_scaling_b':[]}
    keys = set(df['subject'])
    sz = len(keys)
    for k in keys:
        print k
        genuine = list(df[df[subject]==k][array])
        covv = get_covariance_matrix_from_two_observations(genuine[sz+1], genuine[sz+2])
        for_scaling_a = genuine[sz+1]
        for_scaling_b = genuine[sz+2]
        for j in range(sz-1):
            m['a'].append(genuine[j])
            m['b'].append(genuine[j+1])
            m['cov'].append(covv)
            m['for_scaling_a'].append(for_scaling_a)
            m['for_scaling_b'].append(for_scaling_b)
            m[target].append(0)

        genuine = list(df[df[subject]==k].sample(sz-1)[array])
        intruder = list(df[df[subject]!=k].sample(sz-1)[array])

        for a,b in zip(genuine, intruder):
            m['a'].append(a)
            m['b'].append(b)
            m['cov'].append(covv)
            m['for_scaling_a'].append(for_scaling_a)
            m['for_scaling_b'].append(for_scaling_b)
            m[target].append(1)


    bl= pd.DataFrame(m)
    add_distnces_cols(bl)
    print bl[[target]+metrics].corr()

    return bl

def create_as_in_article_data_set(df):
    add_array_col_to_df(df)
    m = df_to_dict(df)
    res = {subject:[], target:[], 'a':[], 'b':[]}
    for k in m.keys():
        pass

metrics = [
    'euclidean',
    'mahalanobis',
    'cityblock',
    'euclidean_scaled',
    'mahalanobis_scaled',
    'cityblock_scaled'
]

def get_scaled_euclidian(a,b, for_scaling_a, for_scaling_b):
    arr = np.array([for_scaling_a, for_scaling_b])
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s==0]=0.001

    a = (a-m)/s
    b = (b-m)/s

    return euclidean(a,b)


def get_scaled_cityblock(a,b, for_scaling_a, for_scaling_b):
    arr = np.array([for_scaling_a, for_scaling_b])
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s==0]=0.001

    a = (a-m)/s
    b = (b-m)/s

    return cityblock(a,b)

def add_distnces_cols(bl):
    bl['euclidean'] = bl.apply(lambda s: euclidean(s['a'], s['b']), axis=1)
    bl['cityblock'] = bl.apply(lambda s: cityblock(s['a'], s['b']), axis=1)
    bl['mahalanobis'] = bl.apply(lambda s: mahalanobis(s['a'], s['b'], s['cov']), axis=1)

    bl['euclidean_scaled'] = \
        bl.apply(lambda s: get_scaled_euclidian(s['a'], s['b'], s['for_scaling_a'], s['for_scaling_b']), axis=1)
    bl['cityblock_scaled'] = \
        bl.apply(lambda s: get_scaled_cityblock(s['a'], s['b'], s['for_scaling_a'], s['for_scaling_b']), axis=1)
    bl['mahalanobis_scaled'] = \
        bl.apply(lambda s: mahalanobis(s['a'], s['b'], s['cov'])/mahalanobis(s['for_scaling_a'], s['for_scaling_b'], s['cov']), axis=1)
