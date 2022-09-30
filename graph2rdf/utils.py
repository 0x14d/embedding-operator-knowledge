import pandas as pd


def get_graph_from_pickle_files(folder_path='data/results3/basic_using_DistMult_head-True/embedding/graph_embeddings/'):
    """reads edge and vertecy data from pickle files and returns these as pandas Dataframe.
       Additionally the 'source' and 'target' column is filtered out from the edge table

    Args:
        folder_path (str, optional): folder path with the pickle files edges.pkl and verts.pkl.
        Defaults to 'data/results3/basic_using_DistMult_head-True/embedding/graph_embeddings/'.

    Returns:
        pd.Dataframe, pd.Dataframe: edge table and metadata of the verticies
    """
    edges = pd.read_pickle(folder_path + 'edges.pkl')
    metadata = pd.read_pickle(folder_path + 'verts.pkl')
    return edges[['source', 'target']], metadata


def get_embeddings_from_tsv_files(folder_path='data/results3/basic_using_DistMult_head-True/embedding/graph_embeddings/'):
    """reads the embedding data and the names from the embeddings of the tsv files

    Args:
        folder_path (str, optional): folder path with the tsv files skg_embeddings.tsv and skg_metadata.tsv
        Defaults to 'data/results3/basic_using_DistMult_head-True/embedding/graph_embeddings/'.

    Returns:
        _type_: _description_
    """
    embeddings = pd.read_csv(
        folder_path + 'skg_embeddings.tsv', sep='\t', names=[i for i in range(0, 48)])
    emb_meta = pd.read_csv(folder_path + 'skg_metadata.tsv', sep='\t')
    return embeddings, emb_meta
