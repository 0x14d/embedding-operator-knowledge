from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# We have to use OrdinalEncoder for label encoding
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


class LabelEncoderForColumns:
    """
    Utility functions to wrap an OrdinalEncoder trained for multiple columns
    """
    _label_encoder: OrdinalEncoder = None
    _columns: List[str] = []
    _column_transformer: ColumnTransformer = None

    def __init__(self, label_encoder: OrdinalEncoder, columns: List[str], column_transformer: ColumnTransformer = None):
        """

        :param label_encoder: Fitted OrdinalEncoder
        :param columns:
        """
        self._label_encoder = label_encoder
        self._columns = columns
        self._column_transformer = column_transformer

    def transform(self, str_to_transform: str, column_to_transform: str) -> int:
        """
        wraps the OrdinalEncoder transform. Constructing input array is necessary since we have one OrdinalEncoder that was trained on all columns -> we need to adjust the input accordingly
        :param str_to_transform:
        :param column_to_transform:
        :return:
        """
        index = self._columns.index(column_to_transform)
        input_array = [[cat[0] for cat in self._label_encoder.categories_]]
        input_array[0][index] = str_to_transform
        return self._label_encoder.transform(input_array)[0][index]

    def inverse_transform(self, int_to_transform: int, column_to_transform: str) -> str:
        """
        wraps the OrdinalEncoders' inverse_transform int -> str
        :param int_to_transform:
        :param column_to_transofrm:
        :return:
        """
        index = self._columns.index(column_to_transform)
        input_array = [[0]*len(self._columns)]
        input_array[0][index] = int_to_transform
        return self._label_encoder.inverse_transform(input_array)[0][index]

    def prepare_array(self, column_to_transform: str, value: Union[int, str]) -> Tuple[np.array, int]:
        index = self._columns.index(column_to_transform)
        input_array = [cat[0] for cat in self._label_encoder.categories_]
        input_array[0][index] = value
        return input_array, index

    def transformdf(self, X: pd.DataFrame, columns: List):
        _X = self._column_transformer.transform(X)
        update_dataframe = pd.DataFrame(_X, columns=columns)
        X.update(update_dataframe)
        X[columns] = X[columns].astype(np.int64)
        return X


class OrdinalArrayEncoder(OrdinalEncoder):
    """Wrapper of Ordinal Encoder to encode numpy arrays instead of dataframes to circumvent UserWarning: X does not have valid feature names, but OrdinalEncoder was fitted with feature name"""

    def __init__(self, *, categories="auto", dtype=np.float64, handle_unknown="error", unknown_value=None):
        super().__init__(categories=categories, dtype=dtype,
                         handle_unknown=handle_unknown, unknown_value=unknown_value)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return super().fit(X.to_numpy(), y)
        else:
            return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            return super().fit_transform(X.to_numpy(), y, **fit_params)
        else:
            return super().fit_transform(X, y, **fit_params)


class Preprocessing:
    @staticmethod
    def duplicate_with_noise(X, y, noisiness=0.01, lov=None, r_state=None):
        """
        Various Sci-kit classifiers require there to be at least two samples
        of each class. This function duplicates a sample and adds minor noise
        so that two are available
        :param X: a pd dataframe containing standardised (!) input features
        :param y: a pd dataframe containing class based outputs
        :param noisiness: the maximum value to be added
        :param lov: a dict containing categories. to these columns no noise
        will be applied
        :return: (X, y) tuple
        """
        def add_noise(x):
            """
            Adds normal distributed noise with sigma defined by outer scope
            noiseness
            :param x: the value to noise
            :return: the noised value
            """
            if r_state is None:
                return x + np.random.normal(scale=noisiness)
            else:
                return x + r_state.normal(scale=noisiness)
        X_ = X.copy(deep=True)
        y_ = y.copy(deep=True)
        X_ = X_.apply(add_noise)
        if lov is not None:
            # lov keys are plural (e.g. printers instead of printer). this
            # resets to singular
            X_ = X_.round({k[:-1]: 0 for k in lov.keys()})
            # reset the type away from float
            X_ = X_.astype({k[:-1]: 'int32' for k in lov.keys()})
        return pd.concat([X, X_]), pd.concat([y, y_])

    @staticmethod
    def remove_artifacts(data):
        """
        There might be some artifacts in the data, which we want removed
        :param data: pd dataframe
        :return:
        """
        size_before = data.shape[0]
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        data.reset_index(drop=True)
        size_after = data.shape[0]
        if size_after != size_before:
            print(f"Removed {size_before - size_after} rows from dataframe due "
                  f"to NaN and Inf values\n")
        return data

    @staticmethod
    def transform_rating_linear(data, weights=None):
        """
        Transform the rating to a (weighted) real valued number.
        :param data: pd dataframe
        :param weights: pd series with indices corresponding to categories
        that should be weighted eg.:
         pandas.Series([2, 3], index=['line_misalignment', 'stringing'])
         note that the default weight is one. Thus, line_misalignment is
         deemed twice and stringing three times as important than the other
         classes in the example
        :return:
        """
        categories = ['blobs', 'gaps', 'layer_misalignment', 'layer_separation', 'over_extrusion', 'line_misalignment',
                      'stringing', 'under_extrusion', 'warping', 'poor_bridging']

        temp_df = data.drop(data.columns.difference(categories), axis=1)
        if weights is not None:
            ones_array = np.ones(temp_df.columns.size, dtype=int)
            ones_weights = pd.Series(ones_array, index=temp_df.columns)
            ones_weights.update(weights)
            rating = (temp_df * ones_weights).sum(axis=1)
        else:
            rating = temp_df.sum(axis=1)
        # first calculating the column then assigning prevents SettingWithCopy Warning
        data = data.assign(rating=rating.values)
        return data

    @staticmethod
    def split_DF(data, Y=None):
        """
        remove columns not needed for learning :param data: a pandas.DataFrame with containing all informations
        :param Y: an iterable containing the name of all columns to drop. Note that the columns 'ID' and 'completion'
        are dropped in every case.
        """
        # make sure 'ID' and 'completion' are removed
        _Y = {'ID', 'completion'}
        if Y is not None:
            _Y.update(Y)
        X = data.copy(deep=True)
        # remove _Y
        for r in _Y:
            # if told to drop a column that does not exist, we just continue
            try:
                X.drop(r, axis=1, inplace=True)
            except Exception as e:
                if "not found in axis" in str(e):
                    pass
                else:
                    raise e
        # ID does not contain useful information
        X.reset_index(drop=True)
        return X

    @staticmethod
    def standardise_columns(X, columns=None):
        """
        Standardises the specified columns, if none are specified, none are
        standardised
        :param X: full inputs
        :param columns: to be standardised
        :return:
        """
        if columns is None:
            return X, None
        column_trans = ColumnTransformer([('StandardScaler', StandardScaler(),
                                           columns)],
                                         remainder='drop', n_jobs=-1)
        X_ = column_trans.fit_transform(X)
        X.update(pd.DataFrame(X_, columns=columns))
        return X, column_trans

    @staticmethod
    def label_encoding(X, columns=None):
        """
        transform the categorical values represented by strings in the given columns into integer-values
        :param X: The full dataset
        :param columns: categorical columns to encode with integer-values
        :return: Tuple (DataFrame with transformed values, transformation)
        """
        if columns is None:
            return X, None
        # n_jobs=1: fixes some internal error...
        column_trans = ColumnTransformer([('Label', OrdinalArrayEncoder(dtype=np.float64), columns)], remainder='drop',
                                         n_jobs=-1)
        _X = column_trans.fit_transform(X)
        update_dataframe = pd.DataFrame(_X, columns=columns)
        X.update(update_dataframe)
        X[columns] = X[columns].astype(np.int64)
        return X, column_trans

    @staticmethod
    def one_hot_encoding(X, columns=None):
        """
        One hot encode the specified columns, if none are specified, none are
        encoded. The new columns are added towards the end of the dataframe,
        while the original column is removed
        :param X: full inputs
        :param columns: to be encoded
        :return:
        """
        if columns is None:
            return X, None
        column_trans = ColumnTransformer([('OneHot', OneHotEncoder(),
                                           columns)], sparse_threshold=0,
                                         remainder='drop', n_jobs=-1)
        X_ = column_trans.fit_transform(X)

        # retrieve the list of individual column entries and flatten it
        c = np.concatenate(column_trans.transformers_[0][
            1].categories_).tolist()
        X_ = pd.DataFrame(X_, columns=c)
        X = pd.concat([X, X_], axis=1)
        X.drop(columns, axis=1, inplace=True)
        return X, column_trans

    @staticmethod
    def prepare_X(data, ratings, lists_of_values):
        """
        ------- DEPRECATED ------- Use Transformers and split_X

        Performs some necessary preparations to work with the data
        :param data: the full data as downloaded from AipeDataProvider
        :param ratings: a list of all column names containing ratings
        :param lists_of_values: a dict of all occuring values for various
        categorical inputs
        :return: X
        """
        X = data.copy(deep=True)
        # ratings are part of Y rather than X
        for r in ratings:
            X.drop(r, axis=1, inplace=True)

        # transform string based categories to integers
        parser = Preprocessing.parser(lists_of_values, 'printers')
        X['printer'] = X['printer'].apply(lambda x: parser[x])
        parser = Preprocessing.parser(lists_of_values, 'material_types')
        X['material_type'] = X['material_type'].apply(lambda x: parser[x])
        parser = Preprocessing.parser(lists_of_values, 'material_producers')
        X['material_producer'] = X['material_producer'].apply(
            lambda x: parser[x])
        parser = Preprocessing.parser(lists_of_values, 'material_colors')
        X['material_color'] = X['material_color'].apply(lambda x: parser[x])

        # ID does not contain useful information
        X.drop('ID', axis=1, inplace=True)
        X.drop('completion', axis=1, inplace=True)
        X.reset_index(drop=True)
        return X

    @staticmethod
    def parser(lists_of_values, key):
        """
        ------- DEPRECATED ------- Use sklearn.preprocessing.LabelEncoder

        Creates a parser (dict) that transforms string based categories to int
        :param lists_of_values: dict where all values are contained.
        subentries are dicts for each "type", e.g. printers
        :param key: which key to create a lookup table for
        :return: a lookup table in dict form
        """
        lookup = {}
        i = 0
        # Sorting the entries makes them somewhat deterministicly ordered
        lovk = list(lists_of_values[key])
        lovk.sort()
        for v in lovk:
            lookup[v] = i
            i += 1
        return lookup

    @staticmethod
    def calculate_vif(data):
        """
        Calculates the Variance Inflation Factor to check for multicollinearity
        (Note that the statsmodels.stats.outliers_influence.variance_inflation_factor() implementation exists
        and is equivalent)
        :param data:
        :return:
        """
        # we do the import here as statsmodels is quite large with
        # dependencies and this is a very niche function. tested with v 0.12.0
        import statsmodels.api as sm
        vif_df = pd.DataFrame(columns=['Var', 'Vif'])
        x_var_names = data.columns
        for i in range(0, x_var_names.shape[0]):
            y = data[x_var_names[i]]
            x = data[x_var_names.drop([x_var_names[i]])]
            r_squared = sm.OLS(y, x).fit().rsquared
            vif = round(1 / (1 - r_squared), 2)
            vif_df.loc[i] = [x_var_names[i], vif]
        return vif_df.sort_values(by='Vif', axis=0, ascending=False,
                                  inplace=False)

    @staticmethod
    def concat_material_information(df, lov, columns_to_concatenate=("material_type", "material_producer", "material_color")):
        """
        Given a pandas dataframe, merge all columns that have material information into one column by dropping
        all but one, filling and renaming it accordingly
        Example:
        before:
                 material_type  material_producer  material_color
        first        PLA                Janbex         Green
        second       PETG               Herz           Blue
        after:
                material_information
        first     PLA:Janbex:Green
        second    PETG:Herz:Blue
        :param df: original (immutable) Dataframe to be merged
        :param lov: original (immutable) lov
        :param columns_to_concatenate: labels of the columns to be concatenated
                                       (standard: "material_type", "material_producer", "material_color")
        :returns: Merged (copied) Dataframe and adjusted (copied) lov
        """
        # deep copy of the data frame to preserve immutability
        copy_df = df.copy(True)
        # adjusting the lov
        lovcon = lov.copy()
        lovcon['material_informations'] = str(lovcon.pop('material_types')) + ":" + str(
            lovcon.pop('material_producers')) + ":" + str(lovcon.pop('material_colors'))
        # iteration over all columns to be concatenated
        for label in columns_to_concatenate:
            # drop the old columns
            copy_df.drop(label, axis=1, inplace=True)
            # fill the new column with the information of the old ones
            if 'material_information' in copy_df:
                # in case there is already information in the new column add ":" as a delimiter
                copy_df['material_information'] += ":" + df[label]
            else:
                copy_df['material_information'] = df[label]
        return copy_df, lovcon

    @staticmethod
    def merge_duplicate_params(df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a pandas dataframe, merge all columns that have the exact same
        values, linear dependencies or an offset into one column by dropping
        all but one and renaming it accordingly
        Example:
        before:
                 x   y    z    n    x2
        first    1   2    3    6     1
        second  10  20   30   15    10
        after:
                x,y(*2.0),z(*3.0),n(+5),x2
        first            1
        second          10
        :param df: original (immutable) Dataframe to be merged
        :returns: Merged (copied) Dataframe
        """
        drop_set = set()
        copy_df = df.copy(True)
        # Iterate over all columns
        for index_col in range(df.shape[1]):
            # Save the column to be checked
            col = copy_df.iloc[:, index_col]
            if index_col not in drop_set:
                # Iterate over all columns after col to compare them with col
                for index_othercol in range(index_col + 1, df.shape[1]):
                    # Save the column to be compared
                    other_col = copy_df.iloc[:, index_othercol]
                    # If both columns are equal, rename col to inhabit the
                    # names of both columns and drop the other column
                    if col.equals(other_col):
                        # Generate the new column name for col
                        new_col_name = (col.name + ',' + other_col.name)
                        # Rename col to new_col
                        copy_df.rename(
                            columns={col.name: new_col_name}, inplace=True)
                        # Save the newly named parameter in col
                        col = copy_df.iloc[:, index_col]
                        # Add redundant column indice to drop set
                        drop_set.add(index_othercol)
                    else:  # checking if there is an offset or a linear dependency
                        # continue in case of non offset/linear dependency compatible types
                        if not (pd.api.types.is_numeric_dtype(col) and pd.api.types.is_numeric_dtype(other_col)):
                            continue
                        # Calculate the factor between the first elements of the columns to be compared
                        # but columns can't have a linear dependency (or a factor) if at least one of them is zero
                        if other_col.iloc[0] != 0 and col.iloc[0] != 0:
                            factor = col.iloc[0] / other_col.iloc[0]
                        else:
                            factor = np.nan
                        try:
                            # Calculate the offset between the first elements of the columns to be compared
                            offset = col.iloc[0] - other_col.iloc[0]
                        except TypeError:
                            continue
                        # Calculate the factor between all elements of the columns to be compared
                        factor_col = col.divide(other_col).to_numpy()
                        # linear_dependency = true if all factors are equal
                        linear_dependency = (factor == factor_col).all()
                        # Calculate the offset between all elements of the columns to be compared
                        offset_col = col.subtract(other_col).to_numpy()
                        # offset_dependency = true if all offsets are equal
                        offset_dependency = (offset == offset_col).all()
                        if linear_dependency:
                            # Generate the new column name for col
                            new_col_name = (
                                col.name + ',' + other_col.name + '(*' + str(1 / factor) + ')')
                            # Rename col to new_col
                            copy_df.rename(
                                columns={col.name: new_col_name}, inplace=True)
                            # Save the newly named parameter in col
                            col = copy_df.iloc[:, index_col]
                            # Add redundant column indice to drop set
                            drop_set.add(index_othercol)
                        elif offset_dependency:
                            # Generate the new column name for col
                            if offset > 0:
                                new_col_name = (
                                    col.name + ',' + other_col.name + '(-' + str(offset) + ')')
                            else:
                                new_col_name = (
                                    col.name + ',' + other_col.name + '(+' + str(-offset) + ')')
                            # Rename col to new_col
                            copy_df.rename(
                                columns={col.name: new_col_name}, inplace=True)
                            # Save the newly named parameter in col
                            col = copy_df.iloc[:, index_col]
                            # Add redundant column indice to drop set
                            drop_set.add(index_othercol)
        # Drop the columns with their indices in drop_set
        copy_df.drop(copy_df.columns[list(drop_set)], axis=1, inplace=True)
        return copy_df

    @staticmethod
    def preprocess_data_to_XY(dp, y=None, i_o=True, c_o=False, l_o=True, r_a=True, c_mi=True, m_d=True, s_x=True,
                              encode='one_hot', handle_nans='impute', m_b=False, r_m=False):
        """
        Given a AipeDataProvider instance, this function gets experiments data and preprocesses them to X and Y according to
        the given parameters.
        :param dp: AipeDataProvider instance to import experiment data
        :param y: selected target (all targets if y = None)
        :param i_o: whether or not oneoffs are included(Standard: True)
        :param c_o: whether or not only completed experiments are included(Standard: False)
        :param l_o: whether or not only labelable experiments are included(Standard: True)
        :param r_a: whether or not artifacts are removed(Standard: True)
        :param c_mi: whether or not material information are concatenated(Standard: True)
        :param m_d: whether or not duplicate columns are merged(Standard: True)
        :param s_x: whether or not X is standardised(Standard: True)
        :param encode: encoding to be used (Standard: one_hot)
        :param handle_nans: handles NaNs in according manner
                            {None, remove_col, remove_row, impute, int/float (for replacing)}
        :param m_b: whether or not dfs multiclass ratings are converted into binary ones
        :param r_m: whether or not multicollinearity is removed from X(Standard: False)
        :returns: preprocessed X and Y
        """
        original_df, lok, lov, _, _, _ = dp.get_executed_experiments_data(include_oneoffs=i_o, completed_only=c_o,
                                                                          labelable_only=l_o)
        # completion has one corrupted element in it which is a string instead of a float which can cause crashes
        original_df['completion'] = pd.to_numeric(
            original_df['completion'], errors='coerce')
        if handle_nans is not None:
            if isinstance(handle_nans, int) or isinstance(handle_nans, float):
                original_df, _, _ = Preprocessing.handle_NaNs(
                    original_df, option='replace', default_value=handle_nans)
            else:
                original_df, _, _ = Preprocessing.handle_NaNs(
                    original_df, option=handle_nans)
        if r_a:
            original_df = Preprocessing.remove_artifacts(original_df)
        if c_mi:
            original_df, lov = Preprocessing.concat_material_information(
                original_df, lov)
        X = Preprocessing.split_DF(original_df, list(lok['ratings']))
        if m_d:
            X = Preprocessing.merge_duplicate_params(X)
        if s_x:
            X, _ = Preprocessing.standardise_columns(X,
                                                     [ele for ele in X.columns.tolist() if ele not in [k[:-1] for k in lov.keys()]])
        if encode == 'one_hot':
            X, _ = Preprocessing.one_hot_encoding(
                X, [k[:-1] for k in lov.keys()])
        elif encode == 'label':
            X, _ = Preprocessing.label_encoding(
                X, [k[:-1] for k in lov.keys()])
        if r_m:
            X = X.astype(float)
            X = Preprocessing.remove_multicollinearity(X)
        Y = original_df[lok['ratings']].copy()
        if m_b:
            X = Preprocessing.make_binary(X)
            Y = Preprocessing.make_binary(Y)
        Y = Y.astype(int)
        X = X.astype(float)
        if y is not None:
            if y not in Y.columns.values:
                Exception(
                    f"Illegal target! \nLegal targets: {Y.columns.values}")
            else:
                Y = Y[y]
            if isinstance(Y, pd.Series):
                Y = Y.to_frame(name=y)
        return X, Y

    @staticmethod
    def handle_NaNs(original_df, option, default_value=0.0):
        """
        Given a data frame this function handles all the NaNs inside the data according to the given options.
        :param original_df: original (immutable) data frame.
        :param option: defines what to do with the NaNs in df. {remove_col, remove_row, replace, impute}
        :param default_value: value that NaNs get replaced with if option = replace
        :returns copied df without NaNs and sets (of labels) of columns and (IDs of) rows that had NaNs in them
        """
        copy_df = original_df.copy(True)
        col_drop_set = set()
        row_drop_set = set()
        nans_in_df = False
        # iterate over all columns
        for parameter in original_df.columns.values:
            # check if column contains NaNs     (nans = array of bools)
            nans = original_df[parameter].isnull()
            # column contains at least one NaN
            if nans.sum() > 0:
                # print(f"Number of NaNs in {parameter} = {nans.sum()}")
                nans_in_df = True
                col_drop_set.add(parameter)
                # iterate through nans to add the (rows of) elements that are NaN to the drop set
                for i in range(copy_df.shape[0]):
                    if nans[i]:
                        row_drop_set.add(i)
        if nans_in_df:
            if option == 'impute':
                # create, fit the imputer and use it to transform the NaNs to numbers
                imp = SimpleImputer(missing_values=np.nan,
                                    strategy='most_frequent')
                copy_numpy = imp.fit_transform(copy_df)
                # reconvert numpyndarray to pandas data frame
                copy_df = pd.DataFrame(data=copy_numpy, index=list(original_df.index),
                                       columns=original_df.columns.values)
            elif option == 'replace':
                # create, fit the imputer and use it to transform the NaNs to numbers
                imp = SimpleImputer(
                    missing_values=np.nan, strategy='constant', fill_value=default_value)
                copy_numpy = imp.fit_transform(copy_df)
                # reconvert numpyndarray to pandas data frame
                copy_df = pd.DataFrame(data=copy_numpy, index=list(original_df.index),
                                       columns=original_df.columns.values)
            elif option == 'remove_col':
                # drop all columns that got NaNs in them
                copy_df.drop(col_drop_set, axis=1, inplace=True)
            elif option == 'remove_row':
                # drop all rows that got NaNs in them
                copy_df.drop(row_drop_set, axis=0, inplace=True)
                # convert positional index (needed for dropping cols) to the experiment ID
                row_drop_set = original_df['ID'].iloc[list(
                    row_drop_set)].tolist()
        return copy_df, col_drop_set, row_drop_set

    @staticmethod
    def remove_multicollinearity(df, max_vif=5, max_drops=None, own_vif_calc=False):
        """
        Given a data frame this function removes the columns with the highest vif as long as it is higher then max_vif.
        :param df: (immutable) DataFrame to remove multicollinearity from
        :param max_vif: threshold for maximal allowed vif
        :param max_drops: threshold of maximal allowed (column-) drops
        :param own_vif_calc: if true Preprocessing.calculate_vif() is used,
                             otherwise statsmodels.stats.outliers_influence.variance_inflation_factor()
        :returns: shortened dataframe which has no more multicollinearity in it
        """
        # TODO: decide which vif calculation is better (they lead to different drops)
        if own_vif_calc:
            X = df.copy(True)
            vif_X = Preprocessing.calculate_vif(X)
            drops = 0
            while vif_X['Vif'].iloc[0] == 'inf' or vif_X['Vif'].iloc[0] > max_vif:
                if max_drops is not None and drops >= max_drops:
                    break
                print(
                    f"Drop #{drops + 1} with vif {vif_X['Vif'].iloc[0]}: {str(vif_X['Var'].iloc[0])}")
                X.drop(str(vif_X['Var'].iloc[0]), axis=1, inplace=True)
                drops += 1
                vif_X = Preprocessing.calculate_vif(X)
        else:
            X = df.copy(True)
            vifs = np.empty(X.shape[1])
            drops = 0
            for i in range(X.shape[1]):
                vifs[i] = variance_inflation_factor(X.to_numpy(), i)
            while vifs[np.argmax(vifs)] == 'inf' or vifs[np.argmax(vifs)] > max_vif:
                if max_drops is not None and drops >= max_drops:
                    break
                maxindex = np.argmax(vifs)
                print(
                    f"Drop #{drops + 1} with vif {vifs[maxindex]}: {X.columns.values[maxindex]}")
                X.drop(X.columns.values[maxindex], axis=1, inplace=True)
                drops += 1
                vifs = np.empty(X.shape[1])
                for i in range(X.shape[1]):
                    vifs[i] = variance_inflation_factor(X.to_numpy(), i)
        return X

    @staticmethod
    def make_binary(df, true_threshold=1):
        """
        This function iterates through all elements of df and convert those >= true_threshold to True(1)
        and the rest to False(0)
        :param df: (immutable) data frame to be converted
        :param true_threshold: threshold for true(1) - false(0) evaluation
        :returns: (copied) binary data frame
        """
        copy_df = df.copy(True)
        for parameter in copy_df.columns.values:
            for i in range(copy_df.shape[0]):
                if copy_df[parameter].iloc[i] >= true_threshold:
                    copy_df[parameter].iloc[i] = 1
                else:
                    copy_df[parameter].iloc[i] = 0
        return copy_df

    @staticmethod
    def fix_boolean_errors(df, columns):
        copy_df = df.copy(True)
        for col in columns:
            copy_df[col] = copy_df[col].astype('bool')
        return copy_df
