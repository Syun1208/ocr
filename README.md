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
    "text": ["a", "b", "c"],
    "score": [0.98, 0.94, 0.92],
    "time_inference": 0.124324
}
```