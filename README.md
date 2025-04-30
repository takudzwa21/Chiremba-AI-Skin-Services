# Chiremba Skin Disease Detection Service

This is a standalone microservice for skin disease detection and classification, part of the Chiremba AI healthcare platform.

## Features

- Skin disease classification using deep learning
- Support for multiple skin conditions including:
  - Chickenpox
  - Cowpox
  - HFMD
  - Measles
  - Monkeypox
  - Healthy skin detection

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Ensure the model file `SkinNet-23M.h5` is present in the root directory

3. Run the service:
   ```bash
   python main.py
   ```

The service will start on port 8080 by default. You can modify the port by setting the `PORT` environment variable.

## API Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `POST /skindisease_classification`: Main endpoint for skin disease classification
  - Accepts: Image file upload
  - Returns: Prediction with confidence scores for top 3 conditions

## Environment Variables

- `PORT`: Server port (default: 8080)

## Dependencies

See `requirements.txt` for full list of dependencies.
```
