import pandas as pd
import numpy as np

from collections import defaultdict

from scipy.linalg import sqrtm
from scipy.spatial.distance import cityblock, mahalanobis, euclidean
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

subject = 'subject'
array = 'array'
target = 'target'

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

# Dwell time (the duration of time a key is pressed) and
# Flight time (the duration of time between "key up" and the next "key down")

fp = '/home/dpetrovskyi/PycharmProjects/keystroke/DSL-StrongPasswordData.csv'
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


def decorrelation_matrix(m):
    covv = np.cov(m, rowvar=True)
    hol = sqrtm(covv)
    # hol = hol.real
    hol = np.linalg.inv(hol)
    # hol = np.transpose(hol)

    return hol


def manhaten_distance_after_decorelation(x, y, hol):
    x = np.dot(hol, x)
    y = np.dot(hol, y)

    return cityblock(x, y)


def add_array_col_to_df(df):
    cols = dwell_cols + flight_cols
    df['array'] = df.apply(lambda s: list(s[cols]), axis=1)
    for col in df.columns:
        if col not in ['array', 'subject']:
            del df[col]


def df_to_dict(df):
    res = defaultdict(list)
    df.apply(lambda s: res[s['subject']].append(s['array']), axis=1)

    return res


def calculate_manhaten_distance_after_decorelation_for_triple(a, b, c):
    m = np.array([a, b]).reshape(len(a), 2)
    hol = decorrelation_matrix(m)

    return manhaten_distance_after_decorelation(a, c, hol), manhaten_distance_after_decorelation(b, c, hol)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_covariance_matrix_from_observations(obs):
    m = np.array(obs).reshape(len(obs[0]), len(obs))
    return np.cov(m, rowvar=True)


def create_balansed_data_set(genuine_samples_num):
    df = pd.read_csv(fp)
    add_array_col_to_df(df)
    m = {'a': [], 'target': [], 'genuine_samples': []}
    keys = set(df['subject'])
    sz = len(keys)
    for k in keys:
        print k
        genuine = list(df[df[subject] == k][array])
        genuine_samples = genuine[:genuine_samples_num]

        for g in genuine[genuine_samples_num:]:
            m['a'].append(g)
            m['genuine_samples'].append(genuine_samples)
            m[target].append(0)

        other_than_genuine_samples_num = len(genuine)-genuine_samples_num
        intruder = list(df[df[subject] != k].sample(other_than_genuine_samples_num)[array])

        for a in intruder:
            m['a'].append(a)
            m['genuine_samples'].append(genuine_samples)
            m[target].append(1)

    bl = pd.DataFrame(m)
    add_distnces_cols(bl)
    return bl


def create_as_in_article_data_set(df):
    add_array_col_to_df(df)
    m = df_to_dict(df)
    res = {subject: [], target: [], 'a': [], 'b': []}
    for k in m.keys():
        pass


#######################################################
def get_scaled_euclidian(a, genuine_samples):
    arr = genuine_samples
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s == 0] = 0.001

    a = (a - m) / s

    return [euclidean(a, (z - m) / s) for z in genuine_samples]


def get_scaled_cityblock(a, genuine_samples):
    arr = genuine_samples
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s == 0] = 0.001

    a = (a - m) / s

    return [cityblock(a, (z - m) / s) for z in genuine_samples]

def get_scaled_cityblock_and_throw_away_n_outlier_coordinates(a, genuine_samples,n):
    arr = genuine_samples
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s == 0] = 0.001

    a = (a - m) / s

    return [throw_away_n_outliers_cityblock(a, (z - m) / s, n) for z in genuine_samples]


def get_scaled_euclidian_and_throw_away_n_outlier_coordinates(a, genuine_samples,n):
    arr = genuine_samples
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s == 0] = 0.001

    a = (a - m) / s

    return [throw_away_n_outliers_euclidian(a, (z - m) / s, n) for z in genuine_samples]


def throw_away_n_outliers_cityblock(a,b, n):
    d = [abs(x-y) for x,y in zip(a,b)]
    d.sort()
    return np.mean(d[:-n])

def throw_away_n_outliers_euclidian(a,b, n):
    d = [abs(x-y)**2 for x,y in zip(a,b)]
    d.sort()
    return np.sqrt(np.mean(d[:-n]))




# metrics
def mean_euclidian(a, genuine_samples):
    return np.mean([euclidean(a, z) for z in genuine_samples])


def mean_cityblock(a, genuine_samples):
    return np.mean([cityblock(a, z) for z in genuine_samples])


def mean_mahalanobis(a, genuine_samples):
    covv = get_covariance_matrix_from_observations(genuine_samples)
    return np.mean([mahalanobis(a, z, covv) for z in genuine_samples])


def min_euclidian(a, genuine_samples):
    return np.min([euclidean(a, z) for z in genuine_samples])


def min_cityblock(a, genuine_samples):
    return np.min([cityblock(a, z) for z in genuine_samples])


def min_mahalanobis(a, genuine_samples):
    covv = get_covariance_matrix_from_observations(genuine_samples)
    return np.min([mahalanobis(a, z, covv) for z in genuine_samples])


def max_euclidian(a, genuine_samples):
    return np.max([euclidean(a, z) for z in genuine_samples])


def max_cityblock(a, genuine_samples):
    return np.max([cityblock(a, z) for z in genuine_samples])


def max_mahalanobis(a, genuine_samples):
    covv = get_covariance_matrix_from_observations(genuine_samples)
    return np.max([mahalanobis(a, z, covv)for z in genuine_samples])


def mean_scaled_euclidian(a, genuine_samples):
    return np.mean(get_scaled_euclidian(a, genuine_samples))

def mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples,n):
    return np.mean(get_scaled_euclidian_and_throw_away_n_outlier_coordinates(a, genuine_samples, n))


def min_scaled_euclidian(a, genuine_samples):
    return np.min(get_scaled_euclidian(a, genuine_samples))


def max_scaled_euclidian(a, genuine_samples):
    return np.max(get_scaled_euclidian(a, genuine_samples))


def mean_scaled_cityblock(a, genuine_samples):
    return np.mean(get_scaled_cityblock(a, genuine_samples))

def mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples,n):
    return np.mean(get_scaled_cityblock_and_throw_away_n_outlier_coordinates(a, genuine_samples, n))


def min_scaled_cityblock(a, genuine_samples):
    return np.min(get_scaled_cityblock(a, genuine_samples))


def max_scaled_cityblock(a, genuine_samples):
    return np.max(get_scaled_cityblock(a, genuine_samples))


def elastic_net(a, genuine_samples):
    return np.mean([mean_scaled_cityblock(a, genuine_samples),
                    mean_scaled_euclidian(a, genuine_samples)])


#######################################################

metrics = {
    mean_euclidian.__name__: mean_euclidian,
    max_euclidian.__name__: max_euclidian,
    min_euclidian.__name__: min_euclidian,
    mean_cityblock.__name__: mean_cityblock,
    max_cityblock.__name__: max_cityblock,
    min_cityblock.__name__: min_cityblock,
    mean_mahalanobis.__name__: mean_mahalanobis,
    mean_scaled_euclidian.__name__: mean_scaled_euclidian,
    max_scaled_euclidian.__name__: max_scaled_euclidian,
    min_scaled_euclidian.__name__: min_scaled_euclidian,
    mean_scaled_cityblock.__name__: mean_scaled_cityblock,
    max_scaled_cityblock.__name__: max_scaled_cityblock,
    min_scaled_cityblock.__name__: min_scaled_cityblock,
    elastic_net.__name__: elastic_net,
    'scaled_cityblock_without_1_biggest':
        lambda a, genuine_samples: mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples, 1),
    'scaled_cityblock_without_2_biggest':
        lambda a, genuine_samples: mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples, 2),
    'scaled_cityblock_without_3_biggest':
        lambda a, genuine_samples: mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples, 3),
    'scaled_cityblock_without_4_biggest':
        lambda a, genuine_samples: mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples, 4),
    'scaled_cityblock_without_5_biggest':
        lambda a, genuine_samples: mean_scaled_cityblock_and_throw_n_outlier_coordinates(a, genuine_samples, 5),
    'scaled_euclidian_without_1_biggest':
        lambda a, genuine_samples: mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples, 1),
    'scaled_euclidian_without_2_biggest':
        lambda a, genuine_samples: mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples, 2),
    'scaled_euclidian_without_3_biggest':
        lambda a, genuine_samples: mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples, 3),
    'scaled_euclidian_without_4_biggest':
        lambda a, genuine_samples: mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples, 4),
    'scaled_euclidian_without_5_biggest':
        lambda a, genuine_samples: mean_scaled_euclidian_and_throw_n_outlier_coordinates(a, genuine_samples, 5)

}


def add_distnces_cols(bl):
    for metric_name, metric in metrics.iteritems():
        print metric_name
        bl[metric_name] = bl.apply(lambda s: metric(s['a'], s['genuine_samples']), axis=1)


def explore_correlation(bl):
    return bl[[target] + metrics.keys()].corr()


def explore_auc(bl):
    for metric_name in metrics.keys():
        auc = roc_auc_score(bl[target], bl[metric_name])
        print '{}: {}'.format(metric_name, auc)

def explore_accuracy_at(bl, at=0.025):
    for metric_name in metrics.keys():
        minn = bl[metric_name].min()
        maxx = bl[metric_name].max()
        bl['tmp'] = (bl[metric_name]-minn)/maxx
        auc = accuracy_score(bl[target], bl['tmp']>at)
        print '{}: {}'.format(metric_name, auc)

def explore_eer(bl):
    for metric_name in metrics.keys():
        auc = eer(bl[target], bl[metric_name])
        print '{}: {}'.format(metric_name, auc)


def eer(y_true, y_score):
    a = zip(y_score, y_true)
    a.sort(key=lambda s: s[0])

    sz = len(y_true)
    pos_num = sum(y_true)
    neg_num = sz - pos_num

    fp=neg_num
    fn=0
    for score,y in a:
        if y==0:
            fp-=1
        else:
            fn+=1

        # print 'fp: {}, fn: {}'.format(fp,fn)

        if fp<=fn:
            break

    return float(fp)/sz