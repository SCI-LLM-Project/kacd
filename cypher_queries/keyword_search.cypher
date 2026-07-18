CALL db.index.fulltext.queryNodes('keyword', $keyword)
YIELD node, score
WHERE node:__Entity__ AND score > 0.5
WITH node, score
ORDER BY score DESC
LIMIT 10
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]->(c:Document)
    WITH c, count(distinct n) as freq
    RETURN c //c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping_internal,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    // filter out nodeCount
    WITH c, c.community_rank as rank, c.weight AS weight, COUNT{(c)<-[:IN_COMMUNITY]-()} AS nodeCount
    WHERE nodeCount > 2
    RETURN c //c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping_internal,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE NOT m IN nodes
    RETURN r //r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels_internal,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__) 
    WHERE m IN nodes
    RETURN r // r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels_internal,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities,
collect {
    UNWIND nodes as n
    RETURN elementId(n) AS nID
} as ids

WITH report_mapping_internal, text_mapping_internal, outsideRels_internal, insideRels_internal, entities, ids,
collect {
    UNWIND report_mapping_internal as t
    RETURN t.summary as t
} as report_mapping, 
collect {
    UNWIND report_mapping_internal as t
    RETURN elementId(t) as t
} as report_mapping_ids, 
collect {
    UNWIND text_mapping_internal as t
    RETURN t.text as t
} as text_mapping, 
collect {
    UNWIND text_mapping_internal as t
    RETURN elementId(t) as t
} as text_mapping_ids, 
collect {
    UNWIND outsideRels_internal as t
    RETURN t.description as t
} as outsideRels, 
collect {
    UNWIND outsideRels_internal as t
    RETURN elementId(t) as t
} as outsideRels_ids, 
collect {
    UNWIND insideRels_internal as t
    RETURN t.description as t
} as insideRels, 
collect {
    UNWIND insideRels_internal as t
    RETURN elementId(t) as t
} as insideRels_ids

// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {idEntities: ids, idRelationships: outsideRels_ids+insideRels_ids, idsReportMapping: report_mapping_ids, idsChunks:text_mapping_ids} AS metadata
