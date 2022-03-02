from source.utils import model_preprocess, svc
from source.search import Graph, drugs

if __name__ == "__main__":
    df = model_preprocess()
    Graph(drugs).between_node_heuristic()
    weights = svc(df)
    Graph(drugs).node_i_to_end_node_heuristic(weights)
