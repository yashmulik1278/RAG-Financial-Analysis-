import os
import csv
import en_core_web_sm  # SpaCy model for NER
from collections import defaultdict

nlp = en_core_web_sm.load()

class KnowledgeGraphBuilder:
    def __init__(self):
        self.entities = defaultdict(list)
        
    def extract_entities(self, text):
        doc = nlp(text)
        seen = set()
        unique_entities = []
        for ent in doc.ents:
            entity = (ent.text, ent.label_)
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        return unique_entities
    
    def extract_relationships(self, text):
        doc = nlp(text)
        seen = set()
        unique_relations = []
        for token in doc:
            if token.dep_ in ("attr", "dobj"):
                subject = doc[token.head.left_edge.i : token.head.right_edge.i].text
                object = doc[token.i : token.right_edge.i].text
                relation = (subject, token.head.text, object)
                if relation not in seen:
                    seen.add(relation)
                    unique_relations.append(relation)
        return unique_relations
    
    def create_nodes(self, entities):
        for entity, label in entities:
            # Instead of creating nodes in Neo4j, we'll collect them for CSV export
            self.entities["nodes"].append({"name": entity, "label": label})
    
    def create_relationships(self, relationships):
        for subject, predicate, obj in relationships:
            # Instead of creating relationships in Neo4j, we'll collect them for CSV export
            self.entities["relationships"].append({
                "source": subject,
                "target": obj,
                "type": predicate,
                "weight": 1  # Default weight
            })
    
    def build_graph(self, documents):
        for doc in documents:
            entities = self.extract_entities(doc)
            relationships = self.extract_relationships(doc)
            
            self.create_nodes(entities)
            self.create_relationships(relationships)
    
    def export_to_csv(self, directory="graph_visualization_files"):
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
    
        # Export nodes to CSV
        nodes_file = os.path.join(directory, "nodes.csv")
        with open(nodes_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["name", "label"])
            writer.writeheader()
            for node in self.entities.get("nodes", []):
                writer.writerow(node)
    
        # Export relationships to CSV
        rels_file = os.path.join(directory, "relationships.csv")
        with open(rels_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["source", "target", "type", "weight"])
            writer.writeheader()
            for rel in self.entities.get("relationships", []):
                writer.writerow(rel)
    
        print(f"Graph data exported to {directory}")