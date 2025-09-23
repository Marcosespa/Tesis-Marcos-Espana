import json
from pathlib import Path
from typing import List

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


