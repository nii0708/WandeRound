import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional


class ChatbotThreadManager:
    def __init__(self, data_file: str = "chat_threads.json"):
        self.data_file = data_file
        self.threads = self.load_threads()

    def load_threads(self) -> Dict[str, Any]:
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def save_threads(self):
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.threads, f, ensure_ascii=False, indent=2)

    def create_thread(self, title: Optional[str] = None) -> str:
        thread_id = str(uuid.uuid4())
        self.threads[thread_id] = {
            "title": title or f"Chat {datetime.now().strftime('%b %d, %H:%M')}",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.save_threads()
        return thread_id

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        thinking_steps: Optional[list] = None,
        geopandas_link: Optional[str] = None,
        map_labels: Optional[dict] = None,
    ):
        if thread_id not in self.threads:
            return
        msg: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if thinking_steps:
            msg["thinking_steps"] = thinking_steps
        if geopandas_link:
            msg["geopandas_link"] = geopandas_link
        if map_labels:
            msg["map_labels"] = map_labels
        self.threads[thread_id]["messages"].append(msg)
        self.threads[thread_id]["updated_at"] = datetime.now().isoformat()
        self.save_threads()

    def get_thread_messages(self, thread_id: str) -> List[Dict]:
        return self.threads.get(thread_id, {}).get("messages", [])

    def get_thread_list(self) -> List[Dict]:
        result = []
        for tid, data in self.threads.items():
            result.append({
                "id": tid,
                "title": data["title"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "message_count": len(data["messages"]),
            })
        return sorted(result, key=lambda x: x["updated_at"], reverse=True)

    def delete_thread(self, thread_id: str):
        if thread_id not in self.threads:
            return
        for msg in self.threads[thread_id]["messages"]:
            if msg.get("geopandas_link"):
                try:
                    os.remove(msg["geopandas_link"])
                except OSError:
                    pass
        del self.threads[thread_id]
        self.save_threads()

    def rename_thread(self, thread_id: str, new_title: str):
        if thread_id in self.threads:
            self.threads[thread_id]["title"] = new_title
            self.threads[thread_id]["updated_at"] = datetime.now().isoformat()
            self.save_threads()
