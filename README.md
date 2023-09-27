# API Endpoint: /ocr

## Request:
```
{
  "image_url": "string",
  "image_base64": "string"
}
```
## Response
```
{
  "text": [
    "line",
    "level",
    "work",
    "or",
    "on"
  ],
  "score": [
    0.9004660248756409,
    0.970832884311676,
    0.9257144927978516,
    0.8817387819290161,
    0.7803182005882263
  ],
  "time_inference": 0.08230304718017578
}
```