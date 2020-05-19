import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def preprocess_input(df_train, df_test,
                     normalize_per_year=False,
                     normalize_labels=True,
                     use_boxcox=False,
                     dim=None):
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])

    if use_boxcox:
        scaler_type = preprocessing.PowerTransformer
    else:
        scaler_type = preprocessing.StandardScaler

    if normalize_per_year:
        train_year_dfs = []
        test_year_dfs = []
        for year, group in X_train.groupby('year').groups.items():
            scaler = scaler_type()
            X = X_train.loc[group]
            scaler.fit(X.values)
            year_train_df = pd.DataFrame(scaler.transform(X.values),
                                 columns=X.columns,
                                 index=X.index)

            Xtst = X_test.query('year == @year')
            year_test_df = pd.DataFrame(scaler.transform(Xtst.values),
                                      columns=Xtst.columns,
                                      index=Xtst.index)

            train_year_dfs.append(year_train_df)
            test_year_dfs.append(year_test_df)

        X_train_transformed = pd.concat(train_year_dfs).values
        X_test_transformed = pd.concat(test_year_dfs).values

    else:
        scaler = scaler_type()
        scaler.fit(X_train.values)
        X_train_transformed = scaler.transform(X_train.values)
        X_test_transformed = scaler.transform(X_test.values)

    if dim:
        pca = PCA(n_components=dim)
        pca.fit(X_train_transformed)
        X_train_transformed = pca.transform(X_train_transformed)
        X_test_transformed = pca.transform(X_test_transformed)

    if normalize_labels:
        label_scaler = scaler.fit(y_train.values.reshape(-1, 1))
        y_train_transformed = label_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    else:
        label_scaler = None
        y_train_transformed = y_train.values

    return torch.tensor(X_train_transformed), \
           torch.tensor(X_test_transformed), \
           torch.tensor(y_train_transformed), \
           torch.tensor(df_test['label'].values), \
           label_scaler


if __name__ == '__main__':
    import os
    from sklearn.model_selection import KFold

    df = pd.read_csv(os.path.join('training_data', 'counties.csv'))
    df['year'] = df['year'].astype(str)
    df['GEOID'] = df['GEOID'].astype(str).apply(lambda x: x.zfill(5))
    df = df.set_index(['year', 'GEOID'])
    df[df < 0] = 0
    df = df.rename(columns={'percent': 'label'})
    for k, (train_index, test_index) in enumerate(KFold(3, shuffle=True).split(df)):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        prepro = preprocess_input(df_train, df_test,
                                  normalize_per_year=True,
                                  normalize_labels=True,
                                  use_boxcox=False,
                                  dim=False)
        train_x, test_x, train_y, test_y, label_scaler = prepro