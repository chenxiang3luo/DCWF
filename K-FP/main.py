import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import multiprocessing as mp
import random
import time
from sklearn.model_selection import train_test_split
from extract import get_features
random.seed(7)
np.random.seed(7)
ginis = []



### Parameters ###
r = 1000  #N/P



def closed_world_acc(neighbors,y_test):
    global MON_SITE_NUM
    # logger.info('Calculate the accuracy...')
    p_c = [0] * MON_SITE_NUM
    tp_c = [0] * MON_SITE_NUM

    tp, p = 0, len(neighbors)
    for trueclass , neighbor in zip(y_test, neighbors):
        p_c[trueclass] += 1
        if len(set(neighbor)) == 1:
            guessclass = neighbor[0]
            if guessclass == trueclass:
                tp += 1
                tp_c[guessclass] += 1


    return tp/p, tp_c, p_c

def open_world_acc(neighbors, y_test,MON_SITE_NUM):
    # logger.info('Calculate the precision...')
    tp, wp, fp, p, n = 0, 0, 0, 0 ,0
    neighbors = np.array(neighbors)
    p += np.sum(y_test < MON_SITE_NUM)
    n += np.sum(y_test == MON_SITE_NUM)
    
    for trueclass, neighbor in zip(y_test,neighbors):
        if len(set(neighbor)) == 1:
            guessclass = neighbor[0]
            if guessclass != MON_SITE_NUM:
                if guessclass == trueclass:
                    tp += 1
                else:
                    if trueclass != MON_SITE_NUM: #is monitored site
                        wp += 1
                        # logger.info('Wrong positive:{},{}'.format(trueclass,neighbor))
                    else:
                        fp += 1
                        # logger.info('False positive:{},{}'.format(trueclass,neighbor))

    return tp,wp,fp,p,n
    
def kfingerprinting(X_train,X_test,y_train,y_test):
    global ginis
    # logger.info('training...')
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True)
    model.fit(X_train, y_train)
    ginis.append(model.feature_importances_)
#    M = model.predict(X_test)
    # for i in range(0,len(M)):
    #     x = M[i]
    #     label = str(Y_test[i][0])+'-'+str(Y_test[i][1])
    #     logger.info('%s: %s'%(str(label), str(x)))
    acc = model.score(X_test, y_test)
    #logger.info('Accuracy = %.4f'%acc)
    train_leaf = model.apply(X_train)
    test_leaf = model.apply(X_test)
    # print(model.feature_importances_)
#    joblib.dump(model, 'dirty-trained-kf.pkl')
    return train_leaf, test_leaf

def get_neighbor(params):
    train_leaf, test_leaf, y_train, K = params[0],params[1], params[2], params[3]
    atile = np.tile(test_leaf, (train_leaf.shape[0],1))
    dists = np.sum(atile != train_leaf, axis = 1)
    k_neighbors = y_train[np.argsort(dists)[:K]]
    return k_neighbors

def parallel(train_leaf, test_leaf, y_train, K = 1, n_jobs = 16):
    train_leaves = [train_leaf]*len(test_leaf)
    y_train = [y_train]*len(test_leaf)
    Ks = [K] * len(test_leaf)
    pool = mp.Pool(n_jobs)
    neighbors = pool.map(get_neighbor, zip(train_leaves, test_leaf, y_train,Ks))
    return np.array(neighbors)



# dic = np.load('./feature.npy',allow_pickle=True).item()

# X = np.array(dic['feature'])
# y = np.array(dic['label'])

     #   print(X.shape, y.shape)
    #here just want to save the model 
    # train_leaf, test_leaf = kfingerprinting(X,X[:1],y, y[:1])
    # np.save('dirty_tor_leaf.npy',train_leaf)
tt = ['4w']
for j in tt:
    print(j)
    for i in range(6):
        print('Session',i)
        get_features(i,j)
        MON_SITE_NUM = 100+i*20
        OPEN_WORLD = 0
        tp_of_cls = np.array([0]*MON_SITE_NUM)
        p_of_cls = np.array([0]*MON_SITE_NUM)
        per_acc = np.array([0]*MON_SITE_NUM).astype('float64')

        reports = []
        start_time = time.time()
        folder_num = 1
        shot = 5
        query = 95
        # for train_index, test_index in sss.split(X,y):
        # logger.info('Testing fold %d'%folder_num)
        # folder_num += 1 
        # if folder_num > 2:
        #     break

        # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=70*MON_SITE_NUM,stratify=y)
        dic_train = np.load('./feature_train.npy',allow_pickle=True).item()
        dic_test = np.load('./feature_test.npy',allow_pickle=True).item()
        X_train = np.array(dic_train['feature'])
        y_train = np.array(dic_train['label'])
        X_test = np.array(dic_test['feature'])
        y_test = np.array(dic_test['label'])
        # print(X_train.shape)
        # print(X_test.shape)
        train_leaf, test_leaf = kfingerprinting(X_train,X_test,y_train, y_test)
        neighbors  = parallel(train_leaf, test_leaf, y_train, 3)
        if OPEN_WORLD:
            tp,wp,fp,p,n = open_world_acc(neighbors,y_test,MON_SITE_NUM)
            reports.append(( tp,wp,fp,p,n))

        else:
            result, tp_c, p_c = closed_world_acc(neighbors,y_test)
            print(result)
            reports.append(result)
            tp_of_cls += np.array(tp_c)
            p_of_cls += np.array(p_c)
            print('*******************')
            # print('tp_c')
            # print(tp_c)
            # print('*******************')
            # print('p_of_cls')
            # print(p_of_cls)
            print('*******************')

# if OPEN_WORLD:
#     tps ,wps, fps, ps, ns = 0, 0, 0, 0, 0
#     for report in reports:
#         tps += report[0]
#         wps += report[1]
#         fps += report[2]
#         ps  += report[3]
#         ns  += report[4]
#     print("{},{},{},{},{}".format(tps, wps, fps, ps, ns))
# else:
#     print(np.array(reports).mean())
#     print(reports)
#     for i in range(MON_SITE_NUM):
#         if p_of_cls[i] == 0:
#             continue
    
   

