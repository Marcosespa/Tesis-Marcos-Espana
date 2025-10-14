import json
from pathlib import Path
from typing import List, Dict, Any

import weaviate
from weaviate.classes.config import Property, DataType, Configure


def get_client(host: str = "localhost", http_port: int = 8080, grpc_port: int = 50051):
    """Return a connected Weaviate v4 client."""
    client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
    return client


def ensure_schema(client, schema_path: str) -> None:
    """Create the collection if it does not exist using the provided schema JSON.

    schema JSON example:
    {
      "class": "BookChunk",
      "vectorizer": "none",
      "properties": [...]
    }
    """
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    class_name = schema.get("class")
    if not class_name:
        raise ValueError("Schema JSON must include 'class'")

    collections = client.collections.list_all()
    # list_all can return a list of names (str) or objects with .name depending on client version
    names = [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in collections]
    if class_name in names:
        return

    # Map string data types from schema to Weaviate DataType enums
    def map_type(t: str) -> DataType:
        t_lower = t.lower()
        if t_lower in ("text", "string"):
            return DataType.TEXT
        if t_lower in ("int", "integer"):
            return DataType.INT
        if t_lower in ("number", "float", "double"):
            return DataType.NUMBER
        if t_lower in ("bool", "boolean"):
            return DataType.BOOL
        return DataType.TEXT

    props: List[Property] = []
    for p in schema.get("properties", []):
        dtype_list = p.get("dataType") or ["text"]
        dtype = map_type(dtype_list[0])
        props.append(
            Property(
                name=p["name"],
                data_type=dtype,
                index_filterable=p.get("indexFilterable", False),
            )
        )

    # Use deprecated but stable vectorizer_config none to disable auto-vectorization.
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=props,
    )


def ensure_multi_class_schema(client, schema_path: str) -> Dict[str, str]:
    """Create multiple collections from a multi-class schema JSON.
    
    Returns:
        Dict mapping source paths to class names
    """
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    classes = schema.get("classes", [])
    
    if not classes:
        raise ValueError("Multi-class schema JSON must include 'classes' array")
    
    existing_collections = client.collections.list_all()
    existing_names = [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in existing_collections]
    
    # Source to class mapping
    source_to_class = {}
    
    for class_config in classes:
        class_name = class_config.get("class")
        if not class_name:
            continue
            
        # Skip if collection already exists
        if class_name in existing_names:
            print(f"Collection '{class_name}' already exists, skipping...")
            continue
        
        # Map data types
        def map_type(t: str) -> DataType:
            t_lower = t.lower()
            if t_lower in ("text", "string"):
                return DataType.TEXT
            elif t_lower in ("int", "integer"):
                return DataType.INT
            elif t_lower in ("number", "float", "double"):
                return DataType.NUMBER
            elif t_lower in ("bool", "boolean"):
                return DataType.BOOL
            elif t_lower == "text[]":
                return DataType.TEXT_ARRAY
            elif t_lower == "int[]":
                return DataType.INT_ARRAY
            return DataType.TEXT

        props: List[Property] = []
        for p in class_config.get("properties", []):
            dtype_list = p.get("dataType") or ["text"]
            dtype = map_type(dtype_list[0])
            props.append(
                Property(
                    name=p["name"],
                    data_type=dtype,
                    index_filterable=p.get("indexFilterable", False),
                )
            )

        # Create collection
        try:
            client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=props,
            )
            print(f"✅ Created collection: {class_name}")
            
            # Map source directories to classes
            if class_name.startswith("NIST_"):
                if class_name == "NIST_SP":
                    source_to_class["data/chunks/NIST"] = class_name
                elif class_name == "NIST_CSWP":
                    source_to_class["data/chunks/NIST_CSWP"] = class_name
                elif class_name == "NIST_AI":
                    source_to_class["data/chunks/NIST_AI"] = class_name
            else:
                source_to_class[f"data/chunks/{class_name}"] = class_name
                
        except Exception as e:
            print(f"❌ Error creating collection '{class_name}': {e}")
    
    return source_to_class


def get_source_to_class_mapping() -> Dict[str, str]:
    """Get mapping from data source paths to Weaviate class names."""
    return {
        "data/chunks/NIST": "NIST_SP",
        "data/chunks/NIST_CSWP": "NIST_CSWP", 
        "data/chunks/NIST_AI": "NIST_AI",
        "data/chunks/USENIX": "USENIX",
        "data/chunks/MITRE": "MITRE",
        "data/chunks/OWASP": "OWASP",
        "data/chunks/SecurityTools": "SecurityTools",
        "data/chunks/AISecKG": "AISecKG",
        "data/chunks/AnnoCTR": "AnnoCTR"
    }


