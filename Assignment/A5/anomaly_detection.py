import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


class AnomalyDetection():
    
    def scaleNum(self, df, indices):
        """
            Input: $df represents a DataFrame with two columns: "id" and "features"
                   $indices represents which dimensions in $features that need to be standardized        
            Output: Return a new DataFrame that updates "features" column with specified features standarized.
            Write your code!
        """
        new_df = df.copy()
        features_df = new_df['features'].apply(pd.Series)
        for i in indices: 
            fea = features_df.iloc[:, i].astype(float)
            fea_mean = fea.mean()
            fea_std = fea.std()
            features_df.iloc[:, i] = (fea - fea_mean) / fea_std
        features_df = features_df.dropna(axis=1)
        features_list = features_df.to_records(index=False).tolist()
        new_df['features'] = features_list
        new_df['features'] = new_df['features'].apply(lambda x: list(x))
        return new_df
        
    def cat2Num(self, df, indices):
        """
            Input: $df represents a DataFrame with two columns: "id" and "features"
                   $indices represents which dimensions in $features are categorical features, 
                    e.g., indices = [0, 1] denotes that the first two dimensions are categorical features.        
            Output: Return a new DataFrame that updates the "features" column with one-hot encoding. 
            Write your code!
        """
        if isinstance(df['features'][0], str):
            df['features'] = df['features'].str.replace('[', '', regex=False).str.replace(']', '', regex=False).apply(lambda x: x.split(','))
        new_df = df.copy()
        features_df = new_df['features'].apply(pd.Series)
        for i in sorted(indices, reverse=True):
            fea = features_df.iloc[:, i]
            col_names = fea.unique()
            fea_df = pd.DataFrame()
            for col in col_names:
                fea_df[col] = (fea == col).astype(int)
            features_df = pd.concat([features_df.iloc[:, :i], fea_df, features_df.iloc[:, i+1:]], axis=1)    
        new_df['features'] = features_df.apply(lambda row: row.tolist(), axis=1)
        return new_df
            
    def detect(self, df, k, t):
        """
            Input: $df represents a DataFrame with two columns: "id" and "features"
                $k is the number of clusters for K-Means
                $t is the score threshold
            Output: Return a new DataFrame that adds the "score" column into the input $df and then
                    removes the rows whose scores are smaller than $t.  
            Write your code!
        """
        new_df = df.copy()
        X = np.array(new_df['features'].tolist())
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        new_df['labels'] = kmeans.labels_
        label_stat = new_df['labels'].value_counts().reset_index()
        label_stat.columns = ['labels', 'n']
        n_max = label_stat['n'].max()
        n_min = label_stat['n'].min()
        label_stat['score'] = (n_max - label_stat['n']) / (n_max - n_min)
        new_df = new_df.merge(label_stat, on='labels').drop(columns=['labels', 'n'])
        return new_df[new_df['score'] >= t]
    
 
if __name__ == "__main__":
    # toy dataset
    # data = [(0, ["http", "udt", 4]), \
    #         (1, ["http", "udf", 5]), \
    #         (2, ["http", "tcp", 5]), \
    #         (3, ["ftp", "icmp", 1]), \
    #         (4, ["http", "tcp", 4])]
    # df = pd.DataFrame(data=data, columns = ["id", "features"])

    # logs-features-sample dataset
    df = pd.read_csv('logs-features-sample.csv').set_index('id')
    
    ad = AnomalyDetection()
    
    df1 = ad.cat2Num(df, [0,1])
    print(df1.head())
    
    # df2 = ad.scaleNum(df1, [6]) # for toy
    df2 = ad.scaleNum(df1, list(range(len(df1['features'][0]) - len(df['features'][0]) + 2, len(df1['features'][0])))) # general
    print(df2.head())

    # df3 = ad.detect(df2, 2, 0.9) # for toy
    df3 = ad.detect(df2, 8, 0.97) # for logs-features-sample
    print(df3)
    