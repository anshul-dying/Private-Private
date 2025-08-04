from fastapi import FastAPI
from api.routes import documents, queries, analytics

app = FastAPI(title="Bajaj Finserv Hackathon")

app.include_router(documents.router, prefix="/api/v1")
app.include_router(queries.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")