from vertexai.preview.language_models import TextEmbeddingModel

# JSONL file must be uploaded to bucket ahead of time
# output folder must exist in bucket

# JSONL_FILE_GS_URI = "gs://irad_gcp_embeddings/sampleTweets.jsonl"
# OUTPUT_FOLDER_GS_URI = "gs://irad_gcp_embeddings/"
JSONL_FILE_GS_URI = "gs://irad_gcp_embeddings/cleanedEmails.jsonl"
OUTPUT_FOLDER_GS_URI = "gs://irad_gcp_embeddings/"

textembedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
batch_prediction_job = textembedding_model.batch_predict(
  dataset=[JSONL_FILE_GS_URI],
  destination_uri_prefix=OUTPUT_FOLDER_GS_URI,
)
print(batch_prediction_job.display_name)
print(batch_prediction_job.resource_name)
print(batch_prediction_job.state)
