
from pandas import DataFrame
from iribaker import to_iri

def add_uri_to_embedding(embeddings: DataFrame, metadata: DataFrame, data_uri: str) -> DataFrame:
    embeddings_dict = {}
    for index, node_keys in enumerate(metadata['name']):
        embeddings_dict[to_iri(data_uri + node_keys)] = embeddings.iloc[index].tolist()

    columns_emb = [i for i in range(0, 48)]
    embeddings_df =  DataFrame.from_dict(embeddings_dict, orient='index', columns=columns_emb)
    embeddings_df.index.name = "iri"
    return embeddings_df

def save_embedding_with_uri_as_csv(embedding_with_csv: DataFrame, filename='embeddings.csv') -> None:
    csv_str = embedding_with_csv.to_csv(sep=',', header=True)
    with open(filename, 'w') as of:
        of.write(csv_str)