MATCH (e:`__Entity__`)
WHERE size(e.id) > $min_length
WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
WHERE count > 1
UNWIND nodes AS node
// Add text distance
WITH distinct
[n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
WHERE size(intermediate_results) > 1
WITH collect(intermediate_results) AS results
// combine groups together if they share elements
UNWIND range(0, size(results)-1, 1) as index
WITH results, index, results[index] as result
WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
    CASE WHEN index <> index2 AND
	size(apoc.coll.intersection(acc, results[index2])) > 0
	THEN apoc.coll.union(acc, results[index2])
	ELSE acc
    END
)) as combinedResult
WITH distinct(combinedResult) as combinedResult
// extra filtering
WITH collect(combinedResult) as allCombinedResults
UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
WHERE x <> combinedResultIndex
AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
)
RETURN combinedResult
