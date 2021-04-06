
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCleaner(object):
    
    def __init__(self, dataset):
        
        # Define the labels of the dataset
        labels =  ['time', 'length', 'circulation_lane',
                                                    'speed', 'weight', 'axles',
                                                    'temperature', 'moistness', 'precipitation_type',
                                                    'precipitation_intensity', 'wind_direction', 'wind_speed',
                                                                                                         'lane_state', 'accident']
        
        # Read the excel dataset using the pre-made labels
        dataset = pd.read_excel(dataset, names=labels)
        
        # Replace columns with whitespaces with NaN numpy data
        dataset.replace(' ', np.nan, inplace=True)
        # Drop the rows that have NaN columns
        dataset.dropna(inplace=True)
        
        # Drop date column
        dataset.drop(['time'], axis=1, inplace=True)

        # For every categorical column (excluding accident), use one-hot encoding
        for column in dataset.select_dtypes(include=['object']):
            if  column != 'accident':
                dataset = pd.concat([dataset, pd.get_dummies(dataset[column], prefix=column)],axis=1)
                dataset.drop([column], axis=1, inplace=True)
        
        # Replace the 'Yes/No' notation to 1/0
        dataset.replace(to_replace='Yes', value=1, inplace=True)
        dataset.replace(to_replace='No', value=0, inplace=True)

        df_datos_sel = self.__select_datos(dataset)

        datos_sel_y = df_datos_sel['accident'].to_numpy().reshape(len(df_datos_sel), 1)
        datos_sel_x = df_datos_sel.drop(['accident'], axis=1).to_numpy()

        datos_sel_x = self.__normaliza_datos(datos_sel_x)

        self.train_x, self.train_y, \
        self.valid_x, self.valid_y, \
        self.test_x, self.test_y = \
            self.__split_datos(datos_sel_x, datos_sel_y)

    def __select_datos(self, dataset):
        df_1 = dataset.loc[dataset['accident'] == 1]
        df_2 = dataset.loc[dataset['accident'] == 0]
        
        n1 = len(df_1)
        n2 = len(df_2)
        
        df = df_1.append(df_2.sample(n=n1)) if  min(n1, n2) == n1 else df_2.append(df_1.sample(n=n2))

        return df.sample(frac=1)

    def __normaliza_datos(self, datos_x):
        medias = np.mean(datos_x, axis=0)
        desviaciones = np.std(datos_x, 0)

        datos_norm_X = 2 * (datos_x - medias) / desviaciones
        return datos_norm_X

    def __split_datos(self, datos_x, datos_y):

        train_X, test_X, train_Y, test_Y = train_test_split(datos_x, datos_y, test_size=0.025, random_state=1)
        train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.03, random_state=1)

        return train_X, train_Y, val_X, val_Y, test_X, test_Y

