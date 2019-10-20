from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import math

output_folder = '../output/BasicAnalysis/'

def show_covariance(matrix, show=False, filepath=None):
    '''
    Show correlation matrix
    '''
    df = matrix
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(matrix)

    corr = df.cov()
    plt.clf()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    if show:
        plt.show()
    else:
        plt.savefig(output_folder + 'CovarianceMatrix.png') if filepath is None else plt.savefig(filepath)
    plt.close()

def show_correlation(matrix, show=False, file_path=None):
    '''
    Show correlation matrix
    '''
    df = matrix
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(matrix)

    corr = df.corr()
    plt.clf()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    if show:
        plt.show()
    else:
        plt.savefig(output_folder + 'CorrelationMatrix.png') if file_path is None else plt.savefig(file_path)
    plt.close()

def show_absolute_correlation(matrix, show=False, file_path=None):
    '''
    Show correlation matrix
    '''
    df = matrix
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(matrix)

    corr = df.corr().abs()
    plt.clf()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    if show:
        plt.show()
    else:
        plt.savefig(output_folder + 'AbsoluteCorrelationMatrix.png') if file_path is None else plt.savefig(file_path)
    plt.close()

def show_standard_deviation(matrix, show=False, file_path=None):
    df = matrix
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(matrix)

    m = np.std(df.values, 0)
    plt.clf()
    plt.bar(df.columns, height=m)
    if show:
        plt.show()
    else:
        plt.savefig(output_folder + 'StandardDeviationBar.png') if file_path is None else plt.savefig(file_path)
    plt.close()

def show_standard_deviation_matrix(matrix, shape, show=False, file_path=None):
    df = matrix
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(matrix)

    m = np.std(df.values, 0)
    m = m.reshape(shape[0], shape[1])
    plt.clf()
    sns.heatmap(m, xticklabels=m, yticklabels=m)
    if show:
        plt.show()
    else:
        plt.savefig(output_folder + 'StandardDeviationMatrix.png') if file_path is None else plt.savefig(file_path)
    plt.close()

def pca(matrix, no_dimentions):
    '''
    You can set no_dimentions as the matrix.shape[1], and calculate the
    variance of the output, to determin how many dimentions should be stay.
    '''
    import pandas as pd
    from sklearn.decomposition import PCA

    matrix0 = matrix
    if isinstance(matrix, pd.DataFrame):
        matrix0 = matrix0.as_matrix()
    pca = PCA(n_components=no_dimentions)
    pca.fit(matrix0)
    return pca.transform(matrix0)


def kernel_pca(matrix, kernel='linear', gamma=None, degree=3, coef0=1,
               kernel_params=None, alpha=1.0,
               fit_inverse_transform=False,
               eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False,
               random_state=None, copy_X=True, n_jobs=1):
    '''
    Still in testing stage
    '''
    import pandas as pd
    from sklearn.decomposition import KernelPCA, PCA
    matrix0 = matrix
    if isinstance(matrix, pd.DataFrame):
        matrix0 = matrix0.as_matrix()

    kpca = KernelPCA(kernel=kernel,
                     gamma=gamma,
                     degree=degree,
                     coef0=coef0,
                     kernel_params=kernel_params,
                     alpha=alpha,
                     fit_inverse_transform=fit_inverse_transform,
                     eigen_solver=eigen_solver,
                     tol=tol,
                     max_iter=max_iter,
                     remove_zero_eig=remove_zero_eig,
                     random_state=random_state,
                     copy_X=copy_X,
                     n_jobs=n_jobs)
    X_kpca = kpca.fit_transform(matrix0)
    # X_back = kpca.inverse_transform(X_kpca) # reverse back
    return X_kpca

if __name__ == '__main__':
    # df = pd.read_csv('../resource/diabetes.csv', index_col=False, header = None, names = [i for i in range(61)])
    df = pd.read_csv('../resource/diabetes.csv', index_col=False)
    df = df[df.columns[0:8]].values


    pca_standard_div = pca(df, 8)

    show_standard_deviation(pca_standard_div,show=False, file_path = output_folder + 'pca_standard_div_bar.png')

    # df = pca_standard_div
    # if not isinstance(df, pd.DataFrame):
    #     df = pd.DataFrame(pca_standard_div)
    #
    # m = np.std(df.values, 0)
    # cumulative_m = np.cumsum(m/sum(m))
    # plt.clf()
    # plt.bar(df.columns, height=cumulative_m)
    # plt.savefig(output_folder + 'pca_cumulative_standard_div_bar.png')


    # show_correlation(df.loc[:, 0:400])
    # show_absolute_correlation(df.loc[:, 0:400])
