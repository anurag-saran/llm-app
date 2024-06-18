from datetime import datetime

import uvicorn
from fastapi import FastAPI

from api.api_router import api_router


app = FastAPI()

app.include_router(api_router)


@app.get("/inference/privacy/health")
def health_check():
    return {
        "health": "ok",
        "timestamp": datetime.now().isoformat()
    }

# TODO: Don't add any routes here


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
