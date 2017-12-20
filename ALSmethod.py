import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from sklearn import linear_model
import os
import gc
import ggplot as gg



def als(rating_matrix, prepare_path, prepare_name, transpose_prepare_name, init_user_matrix=None, init_item_matrix=None,\
        factor_num=200, method='l2', lambda_user=0.001, lambda_item=0.001, \
        iteration_num=100, user_loop_num=500, item_loop_num=500, rmse_loop_num=3):
    # 如果p或者q为空值的话，分解原始矩阵作为初始值
    if((init_user_matrix==None)|(init_item_matrix==None)):
        init_user_matrix, init_item_matrix = decomposed_rating_matrix(rating_matrix, factor_num)
    with h5py.File('latentFactor.hdf5', 'w') as latent_factor_file:
        latent_factor_file.create_dataset('userLatentFactor', data=init_user_matrix)
        latent_factor_file.create_dataset('itemLatentFactor', data=init_item_matrix)
    with h5py.File(prepare_path, 'r') as prepare_y, \
            h5py.File('latentFactor.hdf5', 'r+') as latent_factor_file:
        y = prepare_y[prepare_name]
        transpose_y = prepare_y[transpose_prepare_name]
        user_latent_factor_matrix = latent_factor_file['userLatentFactor']
        item_latent_factor_matrix = latent_factor_file['itemLatentFactor']
        user_num, item_num = y.shape
        user_loop_interval = np.linspace(0, user_num, user_loop_num+1, dtype=np.int)
        item_loop_interval = np.linspace(0, item_num, item_loop_num+1, dtype=np.int)
        est_rmse = [1000]
        # if(method=='l2'):
        #     for iter in range(iteration_num):
        #         for i, j in enumerate(user_loop_interval[:-1]):
        #             next_node = user_loop_interval[i+1]
        #             user_latent_factor_matrix[j:next_node, ] = ridge_reg(item_latent_factor_matrix[:], y[j:next_node, ].transpose(), lambdas=lambda_user)
        #         for i, j in enumerate(item_loop_interval[:-1]):
        #             next_node = item_loop_interval[i+1]
        #             item_latent_factor_matrix[j:next_node, ] = ridge_reg(user_latent_factor_matrix[:], y[:, j:next_node], lambdas=lambda_item)
        #         new_rmse = get_RMSE(rating_matrix, user_latent_factor_matrix[:], item_latent_factor_matrix[:], rmse_loop_num)
        #         est_rmse.append(new_rmse)
        #         print('this is the %s th iteration and the change of RMSE is %s:' %(iter, np.abs(new_rmse-est_rmse[iter])))
        if (method == 'l2'):
            for iter in range(iteration_num):
                for i, j in enumerate(user_loop_interval[:-1]):
                    next_node = user_loop_interval[i + 1]
                    user_latent_factor_matrix[j:next_node, ] = ridge_reg(item_latent_factor_matrix[:],
                                                                         y[j:next_node, ].transpose(),
                                                                         lambdas=lambda_user)
                for i, j in enumerate(item_loop_interval[:-1]):
                    next_node = item_loop_interval[i + 1]
                    item_latent_factor_matrix[j:next_node, ] = ridge_reg(user_latent_factor_matrix[:],
                                                                         transpose_y[j:next_node, ].transpose(),
                                                                         lambdas=lambda_item)
                # calculate the RMSE
                new_rmse = get_RMSE(rating_matrix, user_latent_factor_matrix[:], item_latent_factor_matrix[:],
                                    rmse_loop_num)
                est_rmse.append(new_rmse)
                print('this is the %s th iteration ,the RMSE is: %s and the change of RMSE is: %s' % (
                iter, new_rmse, (new_rmse - est_rmse[iter])/est_rmse[iter]))
        if(method=='l1'):
            for iter in range(iteration_num):
                for i, j in enumerate(user_loop_interval[:-1]):
                    next_node = user_loop_interval[i + 1]
                    user_latent_factor_matrix[j:next_node, ] = lasso_reg(item_latent_factor_matrix[:],
                                                                         y[j:next_node, ].transpose(),
                                                                         lambdas=lambda_user)
                for i, j in enumerate(item_loop_interval[:-1]):
                    next_node = item_loop_interval[i + 1]
                    item_latent_factor_matrix[j:next_node, ] = lasso_reg(user_latent_factor_matrix[:],
                                                                         y[:, j:next_node], lambdas=lambda_item)
                # calculate the RMSE
                new_rmse = get_RMSE(rating_matrix, user_latent_factor_matrix[:], item_latent_factor_matrix[:],
                                    rmse_loop_num)
                est_rmse.append(new_rmse)
                print('this is the %s th iteration and the change ratio of RMSE is %s' % (
                iter, (new_rmse - est_rmse[iter])/est_rmse[iter]))
        else:
            print('no this method')
        return est_rmse

def decomposed_rating_matrix(sparse_matrix, k):
    u, s, vt = svds(sparse_matrix, k)
    return u.dot(np.diag(s)), vt.transpose()

def ridge_reg(X , y, lambdas):
    reg = linear_model.Ridge(alpha=lambdas, fit_intercept=False)
    reg.fit(X, y)
    return reg.coef_

def lasso_reg(X, y, lambdas):
    reg = linear_model.Lasso(alpha=lambdas, fit_intercept=False)
    reg.fit(X, y)
    return reg.coef_

#supposed p and q are small enough to fit the memory
def get_RMSE(rating_matrix, user_matrix, item_matrix, loop_num):
    count = len(rating_matrix.data)
    #m, n = rating_matrix.shape
    result = 0
    interval = np.linspace(0, count, num=loop_num+1, dtype=np.int)
    for i, j in enumerate(interval[:-1]):
        next_node = interval[i+1]
        part_row_index = rating_matrix.row[j:next_node]
        part_col_index = rating_matrix.col[j:next_node]
        result+= calcu_RMSE(rating_matrix.data[j:next_node], np.sum(np.multiply(user_matrix[part_row_index, ], item_matrix[part_col_index, ]), axis=1))
    return np.sqrt(result/count)

def calcu_RMSE(observed_rating, estimated_rating):
    return np.sum(np.square(observed_rating-estimated_rating))

def graph(y):
    data = pd.DataFrame({'iteration':list(range(len(y))), 'RMSE':y})
    p = gg.ggplot(gg.aes(x='iteration', y='RMSE'), data=data)+gg.geom_point()+gg.geom_line()
    return p

if __name__=='__main__':
    os.chdir('F:/Data/last.FM/reg_data')
    rating_data = pd.read_csv('output.csv')
    row = rating_data['userID'].get_values()
    col = rating_data['artistID'].get_values()
    data = rating_data['weight'].get_values()

    rating_matrix = coo_matrix((data, (row, col)), dtype=np.float)
    del rating_data, row, col, data
    gc.collect()

    # y_observed = pd.read_csv('y_observed.csv')
    # row = y_observed['userID'].get_values()
    # col = y_observed['artistID'].get_values()
    # data = y_observed['y'].get_values()
    # y_observed_matrix = coo_matrix((data, (row, col)), dtype=np.float)
    # del y_observed, row, col, data
    # gc.collect()

    path = 'F:/Data/last.FM/reg_data/yPrepare.h5'
    name1 = '/yData/y'
    name2 = '/yData/y_trans'
    result = als(rating_matrix, path, name1, name2, factor_num=5, method='l2', iteration_num=100, user_loop_num=1, item_loop_num=5, lambda_user=3, lambda_item=3)
    print(result)
    print(graph(result[1:]))
