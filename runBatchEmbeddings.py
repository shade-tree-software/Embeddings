from vertexai.preview.language_models import TextEmbeddingModel
textembedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
batch_prediction_job = textembedding_model.batch_predict(
  dataset=["gs://irad_gcp_embeddings/sampleTweets.jsonl"],
  destination_uri_prefix="gs://irad_gcp_embeddings/",
)
print(batch_prediction_job.display_name)
print(batch_prediction_job.resource_name)
print(batch_prediction_job.state)
