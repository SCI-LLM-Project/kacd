import helpers
from config import topChunks, topCommunities, topRels, topEntities

path_retrieval_query = helpers.load_query("path_search.cypher")

def get_paths(graph, source, target):
    return graph.query(path_retrieval_query, {"source": source, "target": target, "topEntities": topEntities})

def retrieve_path(graph, var1, var2, debug=False):
    paths = get_paths(graph, var1, var2)
    if debug:
        print(helpers.stringify_paths(paths))
    return helpers.stringify_paths(paths)