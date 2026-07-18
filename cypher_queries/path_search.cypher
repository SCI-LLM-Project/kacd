// Step 1: Get the source nodes
CALL db.index.fulltext.queryNodes('keyword', $source)
YIELD node AS source, score AS sourceScore
WHERE source:__Entity__ AND sourceScore > 0.5
WITH source, sourceScore, count{(source)--()} AS sourceDegree
ORDER BY sourceScore * 0.4 + log10(sourceDegree + 1) * 0.6 DESC
LIMIT $topEntities

// Step 2: Get top 10 nodes with second keyword with high degree 
CALL {
    CALL db.index.fulltext.queryNodes('keyword', $target)
    YIELD node AS target, score AS targetScore
    WHERE target:__Entity__ AND targetScore > 0.5
    WITH target, targetScore, count{(target)--()} AS targetDegree
    RETURN target, targetScore, targetDegree
    ORDER BY targetScore * 0.4 + log10(targetDegree + 1) * 0.6 DESC
    LIMIT $topEntities
}

// Step 3: Find shortest paths between each source and target using Dijkstra's algorithm
CALL gds.shortestPath.dijkstra.stream('graph',
    {
        nodeLabels: ['__Entity__'],
        relationshipTypes: ['IS_ASSOCIATED_WITH', 'CAUSES'],
        sourceNode: source,
        targetNode: target
        //relationshipWeightProperty: 'weights'
    }
)

// Step 4: Reconstruct the exact path, since the path returned by djikstra's excludes relationship info
YIELD nodeIds, path, totalCost
WITH path, totalCost, nodeIds
UNWIND range(0, size(nodeIds)-2) AS i
WITH path, totalCost, nodeIds[i] AS sourceId, nodeIds[i+1] AS targetId
MATCH (c1)<-[:IN_COMMUNITY]-(source)-[r]->(target)-[:IN_COMMUNITY]->(c2)
WHERE id(source) = sourceId AND id(target) = targetId
WITH nodes(path) as path, totalCost, collect({sourceComm: c1.summary, targetComm: c2.summary, sourceId: source.id, sourceDesc: source.description, relationship: type(r), relationshipDesc: r.description, targetId: target.id, targetDesc: target.description}) AS triplets
RETURN path, totalCost, triplets
ORDER BY totalCost ASC
LIMIT 3