
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import Response,StreamingResponse
import asyncio
import json


app = FastAPI(title="My Simple API", version="1.0.0")


async def generate_text():
    """异步生成器，逐步产生文本块"""
    for i in range(10):
        # 模拟生成数据
        yield f"Chunk {i}\n"
        await asyncio.sleep(0.5)  # 模拟耗时操作

async def event_stream():
    for i in range(10):
        yield f"data: {json.dumps({'count': i})}\n\n"
        await asyncio.sleep(1)

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "message": f"Item {item_id}"}

@app.get("/search/")
def search_items(q: str = "", skip: int = 0, limit: int = 10):
    return {"query": q, "skip": skip, "limit": limit}


@app.post("/items/")
def create_item(item: Item):
    total = item.price + (item.tax or 0)
    return {
        "name": item.name,
        "price": item.price,
        "tax": item.tax,
        "total": total,
        "message": "Item created"
    }

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, q: Optional[str] = None):
    return {
        "item_id": item_id,
        "item": item,
        "query": q,
        "message": "Item updated"
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=200)  # 空响应，浏览器不会报错


@app.get("/stream")
async def stream_text():
    return StreamingResponse(generate_text(), media_type="text/plain")


@app.get("/events")
async def sse():
    return StreamingResponse(event_stream(), media_type="text/event-stream")