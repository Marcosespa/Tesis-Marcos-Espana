import weaviate

# Weaviate Python client v4
client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)

print(client.is_ready())

client.close()