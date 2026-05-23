import os
import json
import traceback
import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app import AgentGraph
from thread_manager import ChatbotThreadManager

# Max number of recent messages from a thread to inject as conversation context.
HISTORY_LIMIT = 20

api = FastAPI(title="WandeRound API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = AgentGraph()
_agent.model_choose("DEEPSEEK")
_threads = ChatbotThreadManager()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _load_history_as_messages(thread_id: str, latest_user_message: str) -> List[BaseMessage]:
    """Load thread history and return it as LangChain messages, capped to HISTORY_LIMIT."""
    raw = _threads.get_thread_messages(thread_id) or []
    converted: List[BaseMessage] = []
    for m in raw:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))

    # Frontend saves the new user message before calling /chat/stream, so it
    # should already be the last entry. If not (e.g. save failed), append it.
    if not converted or not (
        isinstance(converted[-1], HumanMessage)
        and converted[-1].content == latest_user_message
    ):
        converted.append(HumanMessage(content=latest_user_message))

    if len(converted) > HISTORY_LIMIT:
        converted = converted[-HISTORY_LIMIT:]
    return converted


async def _run_agent(thread_id: str, message: str):
    state = {
        "messages": _load_history_as_messages(thread_id, message),
        "location": None,
        "geocode_data": None,
        "error": None,
    }
    geopandas_file: Optional[str] = None
    map_labels: Optional[dict] = None
    final_response = ""
    thinking_steps: list = []
    streamed_response = ""
    streaming_nodes = {"usual", "get_summary"}

    try:
        async for ev in _agent.graph.astream_events(state, version="v2"):
            kind = ev.get("event", "")
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""

            if kind == "on_chat_model_stream" and node in streaming_nodes:
                chunk = ev.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "") if chunk is not None else ""
                if content:
                    streamed_response += content
                    yield _sse({"type": "delta", "content": content})
                continue

            if kind == "on_chain_end" and node:
                output = ev.get("data", {}).get("output")
                if not isinstance(output, dict):
                    continue

                if "steps" in output:
                    steps = output.get("steps", [])
                    overpass_instr = output.get("overpassInstructions", [])
                    thinking_steps += steps + overpass_instr
                    yield _sse({"type": "thinking", "steps": steps, "overpass": overpass_instr})

                if "overpassResponses" in output:
                    queries = output.get("overpassResponses", [])
                    thinking_steps += queries
                    yield _sse({"type": "overpass", "queries": queries})

                if "stepCodes" in output:
                    codes = output.get("stepCodes")
                    code = codes[-1] if isinstance(codes, list) else codes
                    if hasattr(code, "content"):
                        code = code.content
                    code_str = str(code)
                    thinking_steps.append(code_str)
                    yield _sse({"type": "code", "content": code_str})

                if "geopandasData" in output:
                    geopandas_file = output.get("geopandasData")

                if "mapLabels" in output:
                    map_labels = output.get("mapLabels")

                if "finalResponse" in output:
                    resp = output.get("finalResponse")
                    if hasattr(resp, "content"):
                        final_response = resp.content
                    elif isinstance(resp, list) and resp:
                        r = resp[-1]
                        final_response = r.content if hasattr(r, "content") else str(r)
                    else:
                        final_response = str(resp) if resp else ""

        if streamed_response:
            final_response = streamed_response
    except Exception as e:
        print("=== AGENT ERROR ===")
        traceback.print_exc()
        print("=== END ERROR ===")
        yield _sse({"type": "error", "content": str(e)})
        yield _sse({"type": "done", "content": f"Error: {e}", "thinking": thinking_steps, "file": None})
        return

    if geopandas_file:
        try:
            gdf = gpd.read_file(geopandas_file).to_crs(epsg=4326)
            geojson = json.loads(gdf.to_json())
            yield _sse({
                "type": "map",
                "geojson": geojson,
                "file": geopandas_file,
                "labels": map_labels,
            })
        except Exception as e:
            print(f"Map conversion error: {e}")

    yield _sse({"type": "response", "content": final_response})
    yield _sse({
        "type": "done",
        "content": final_response,
        "thinking": thinking_steps,
        "file": geopandas_file,
        "labels": map_labels,
    })


# ── Request models ────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    thread_id: str
    message: str

class CreateThreadReq(BaseModel):
    title: Optional[str] = None

class RenameThreadReq(BaseModel):
    title: str

class AddMessageReq(BaseModel):
    role: str
    content: str
    thinking_steps: Optional[list] = None
    geopandas_link: Optional[str] = None
    map_labels: Optional[dict] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@api.post("/api/chat/stream")
def chat_stream(req: ChatReq):
    return StreamingResponse(
        _run_agent(req.thread_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get("/api/threads")
def list_threads():
    return _threads.get_thread_list()


@api.post("/api/threads")
def create_thread(req: CreateThreadReq):
    tid = _threads.create_thread(req.title)
    return {"id": tid}


@api.get("/api/threads/{thread_id}/messages")
def get_messages(thread_id: str):
    return _threads.get_thread_messages(thread_id)


@api.post("/api/threads/{thread_id}/messages")
def add_message(thread_id: str, req: AddMessageReq):
    _threads.add_message(
        thread_id,
        req.role,
        req.content,
        req.thinking_steps,
        req.geopandas_link,
        req.map_labels,
    )
    return {"ok": True}


@api.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    _threads.delete_thread(thread_id)
    return {"ok": True}


@api.patch("/api/threads/{thread_id}")
def rename_thread(thread_id: str, req: RenameThreadReq):
    _threads.rename_thread(thread_id, req.title)
    return {"ok": True}


@api.get("/api/map")
def get_map(file: str):
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="Map file not found")
    try:
        gdf = gpd.read_file(file).to_crs(epsg=4326)
        return json.loads(gdf.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
