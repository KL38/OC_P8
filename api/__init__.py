"""Runtime FastAPI application for the credit scoring model.

Light, focused module: only the transformations needed to turn a single
JSON payload into a 768-feature row the model can score. Heavy aggregation
is precomputed offline by feature_engineering/.
"""
