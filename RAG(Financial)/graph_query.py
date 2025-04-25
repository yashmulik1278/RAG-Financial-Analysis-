from neo4j import GraphDatabase

class GraphQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [dict(record) for record in result]
    
    def find_related_entities(self, entity_name, depth=2):
        query = """
        MATCH (e:Entity {name: $entity})-[r*1..$depth]->(related)
        RETURN related, r
        """
        parameters = {"entity": entity_name, "depth": depth}
        return self.query(query, parameters)
    
    def find_shortest_path(self, start_entity, end_entity):
        query = """
        MATCH (start:Entity {name: $start}), (end:Entity {name: $end})
        CALL apoc.algo.dijkstra(start, end, 'RELATED_TO*', null, 'weight')
        YIELD path, weight
        RETURN path, weight
        """
        parameters = {"start": start_entity, "end": end_entity}
        return self.query(query, parameters)

# Usage
if __name__ == "__main__":
    # Replace with your Aura connection details
    AURA_URI = "neo4j+s://837c9ef4.databases.neo4j.io"
    AURA_USER = "neo4j"
    AURA_PASSWORD = "s0WvgStLpIuw_9xQWVoK3roJY15ESeSmjgB6onDbOno"

    graph = GraphQuery(AURA_URI, AURA_USER, AURA_PASSWORD)
    try:
        results = graph.find_related_entities("your_entity", depth=3)
        print(results)
    finally:
        graph.close()