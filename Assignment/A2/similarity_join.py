import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

        
    def preprocess_df(self, df, cols): 
        new_df = df.copy()
        joinString = ''        
        for col in cols:
            joinString += new_df[col].apply(lambda x: x if pd.notna(x) else '') + ' '
        new_df['joinKey'] = joinString.apply(lambda x: re.split(r'\W+', x.lower())).apply(lambda lst: [i for i in lst if i])        
        return new_df
        
    
    def filtering(self, df1, df2):
        df1 = df1[['id', 'joinKey']].rename(columns={'id': 'id1'})
        df2 = df2[['id', 'joinKey']].rename(columns={'id': 'id2'})
        df1_n = df1.explode('joinKey')
        df2_n = df2.explode('joinKey')

        df_m = df1_n.merge(df2_n, on='joinKey')
        df_keep_ids = df_m.drop_duplicates(subset=['id1', 'id2'])[['id1', 'id2']]

        cand_df = df_keep_ids.merge(df1, on='id1').rename(columns={'joinKey': 'joinKey1'})
        cand_df = cand_df.merge(df2, on='id2').rename(columns={'joinKey': 'joinKey2'})
        cand_df = cand_df[['id1', 'joinKey1', 'id2', 'joinKey2']]
        
        return cand_df

      
    def verification(self, cand_df, threshold):
        new_df = cand_df.copy()
        new_df['jaccard'] = new_df.apply(lambda x: len(set(x['joinKey1']).intersection(set(x['joinKey2']))) / len(set(x['joinKey1']).union(set(x['joinKey2']))), axis=1)        
        return new_df[new_df.jaccard>=threshold]
    
        
    def evaluate(self, result, ground_truth):
        r_lst = [x[0]+'-'+x[1] for x in result]
        a_lst = [x[0]+'-'+x[1] for x in ground_truth]
        t_lst = set(r_lst) & set(a_lst)
        
        precision = len(t_lst) / len(r_lst)
        recall = len(t_lst) / len(a_lst)
        fmeasure = 2 * precision * recall / (precision + recall)
        
        return (precision, recall, fmeasure)

               
    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 
        
        cand_df = self.filtering(new_df1, new_df2)
        print("After Filtering: %d pairs left" %(cand_df.shape[0]))
        
        result_df = self.verification(cand_df, threshold)
        result_df.to_csv('tocheck_result_df_sample.csv', index=False)
        print("After Verification: %d similar pairs" %(result_df.shape[0]))
        
        return result_df
       
        

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    # er = SimilarityJoin("Amazon.csv", "Google.csv")

    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    # ground_truth = pd.read_csv("Amazon_Google_perfectMapping.csv").values.tolist()
    print("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))