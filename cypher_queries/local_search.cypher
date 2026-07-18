WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:MENTIONS]-(c:Document)
    WITH c, count(distinct n) as freq
    RETURN c
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping_internal,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY*]->(c:__Community__)
    // filter out nodeCount
    WITH c, c.community_rank as rank, c.impact_severity_rating AS weight, COUNT{(c)<-[:IN_COMMUNITY]-()} AS nodeCount
    WHERE nodeCount > 1
    RETURN DISTINCT c
    // ORDER BY c.impact_severity_rating DESC
    // ordering by level ascending because leaf communities empirically have shown to be more specific to the variables. Higher level tends to be less focused on the variables.
    // Need to sort first by level, because we might retrieve both parent community reports and their child community reports, 
    // which tend to be nearly identical. So we sort by level to prevent that.
    // Note this was empirically determined
    ORDER BY c.level ASC, c.community_rank DESC, c.impact_severity_rating DESC
    LIMIT $topCommunities
} AS report_mapping_internal,
collect {
    UNWIND nodes as n
    MATCH (n)-[r]-(m:__Entity__)
    WITH r,
        apoc.node.degree(startNode(r)) + apoc.node.degree(endNode(r)) as totalDegree
    ORDER BY totalDegree DESC, r.strength DESC
    LIMIT $topRels
    RETURN {
        start_id: startNode(r).id,
        start_desc: startNode(r).description,
        rel_desc: r.description,
        rel_type: type(r),
        end_id: endNode(r).id,
        end_desc: endNode(r).description,
        degree: totalDegree
    }
} as relationships,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities,
collect {
    UNWIND nodes as n
    RETURN elementId(n) AS nID
} as ids

WITH report_mapping_internal, text_mapping_internal, entities, ids, relationships,
collect {
    UNWIND text_mapping_internal as t
    RETURN t.text as t
} as text_mapping

// We don't have covariates or claims here
RETURN {Entities: entities} AS text, 1.0 AS score, {Chunks: text_mapping, Reports: report_mapping_internal, Relationships: relationships, Entities: entities, idEntities: ids} AS metadata
