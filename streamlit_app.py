import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import uuid
import geopandas as gpd
import branca.colormap as cm
from streamlit_folium import st_folium, folium_static
import app

if "chatbot" not in st.session_state:
    st.session_state.chatbot = app.AgentGraph()


class ChatbotThreadManager:
    def __init__(self, data_file: str = "chat_threads.json"):
        self.data_file = data_file
        self.threads = self.load_threads()

    def load_threads(self) -> Dict[str, Any]:
        """Load threads from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def save_threads(self):
        """Save threads to JSON file"""
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.threads, f, ensure_ascii=False, indent=2)

    def create_thread(self, title: str = None) -> str:
        """Create a new chat thread"""
        thread_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        self.threads[thread_id] = {
            "title": title,
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
        thinking_steps: list = None,
        geopandasData: str = None,
    ):
        """Add a message to a thread"""
        if thread_id in self.threads:
            message_data = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }

            # Add thinking steps if provided (for assistant messages)
            if thinking_steps:
                message_data["thinking_steps"] = thinking_steps

            if geopandasData:
                message_data["geopandas_link"] = geopandasData

            self.threads[thread_id]["messages"].append(message_data)
            self.threads[thread_id]["updated_at"] = datetime.now().isoformat()
            self.save_threads()

    def get_thread_messages(self, thread_id: str) -> List[Dict]:
        """Get all messages from a thread"""
        return self.threads.get(thread_id, {}).get("messages", [])

    def get_thread_list(self) -> List[Dict]:
        """Get list of all threads with metadata"""
        thread_list = []
        for thread_id, thread_data in self.threads.items():
            thread_list.append(
                {
                    "id": thread_id,
                    "title": thread_data["title"],
                    "created_at": thread_data["created_at"],
                    "updated_at": thread_data["updated_at"],
                    "message_count": len(thread_data["messages"]),
                }
            )
        return sorted(thread_list, key=lambda x: x["updated_at"], reverse=True)

    def delete_thread(self, thread_id: str):
        """Delete a thread"""
        if thread_id in self.threads:
            try:
                for message in self.threads[thread_id]["messages"]:
                    if message.get("geopandas_link"):
                        os.remove(message["geopandas_link"])
                        print(message["geopandas_link"])
            except:
                print("No geopandas link found")
            del self.threads[thread_id]
            self.save_threads()

    def rename_thread(self, thread_id: str, new_title: str):
        """Rename a thread"""
        if thread_id in self.threads:
            self.threads[thread_id]["title"] = new_title
            self.threads[thread_id]["updated_at"] = datetime.now().isoformat()
            self.save_threads()


def main():
    st.set_page_config(page_title="Chatbot with Threads", page_icon="💬", layout="wide")

    st.title("💬 Chatbot with Thread Management")

    # Initialize thread manager
    if "thread_manager" not in st.session_state:
        st.session_state.thread_manager = ChatbotThreadManager()

    # Initialize current thread
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None

    # Sidebar for thread management
    with st.sidebar:
        st.header("💾 Chat Threads")

        # New thread button
        if st.button("➕ New Thread", use_container_width=True):
            new_thread_id = st.session_state.thread_manager.create_thread()
            st.session_state.current_thread_id = new_thread_id
            st.rerun()

        st.divider()

        # Thread list
        threads = st.session_state.thread_manager.get_thread_list()

        if threads:
            st.subheader("Recent Threads")

            for thread in threads:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Thread selection button
                    is_current = thread["id"] == st.session_state.current_thread_id
                    if st.button(
                        f"{'🔹' if is_current else '💬'} {thread['title'][:20]}...",
                        key=f"select_{thread['id']}",
                        use_container_width=True,
                        type="primary" if is_current else "secondary",
                    ):
                        st.session_state.current_thread_id = thread["id"]
                        st.rerun()

                with col2:
                    # Rename button
                    if st.button("✏️", key=f"rename_{thread['id']}", help="Rename"):
                        st.session_state[f"rename_mode_{thread['id']}"] = True
                        st.rerun()

                with col3:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{thread['id']}", help="Delete"):
                        st.session_state.thread_manager.delete_thread(thread["id"])
                        if st.session_state.current_thread_id == thread["id"]:
                            st.session_state.current_thread_id = None
                        st.rerun()

                # Rename input
                if st.session_state.get(f"rename_mode_{thread['id']}", False):
                    new_title = st.text_input(
                        "New title:",
                        value=thread["title"],
                        key=f"new_title_{thread['id']}",
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("💾", key=f"save_rename_{thread['id']}"):
                            st.session_state.thread_manager.rename_thread(
                                thread["id"], new_title
                            )
                            st.session_state[f"rename_mode_{thread['id']}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.button("❌", key=f"cancel_rename_{thread['id']}"):
                            st.session_state[f"rename_mode_{thread['id']}"] = False
                            st.rerun()

                # Thread info
                st.caption(
                    f"Messages: {thread['message_count']} | {datetime.fromisoformat(thread['updated_at']).strftime('%m/%d %H:%M')}"
                )
                st.divider()
        else:
            st.info("No threads yet. Create your first thread!")

    # Main chat area
    if st.session_state.current_thread_id is None:
        st.info(
            "👋 Welcome! Create a new thread or select an existing one to start chatting."
        )
        if st.button("🚀 Start Your First Chat"):
            new_thread_id = st.session_state.thread_manager.create_thread(
                "My First Chat"
            )
            st.session_state.current_thread_id = new_thread_id
            st.rerun()
    else:
        # Display current thread title
        current_thread = st.session_state.thread_manager.threads.get(
            st.session_state.current_thread_id, {}
        )
        st.subheader(f"💬 {current_thread.get('title', 'Chat Thread')}")

        # Chat messages container
        messages = st.session_state.thread_manager.get_thread_messages(
            st.session_state.current_thread_id
        )

        # Display chat history
        for message in messages:
            with st.chat_message(message["role"]):
                # Show thinking steps for assistant messages
                if message["role"] == "assistant" and "thinking_steps" in message:
                    with st.expander("🤔 **Show Thinking Process**", expanded=False):
                        for step in message["thinking_steps"]:
                            if step.startswith("\n"):
                                st.markdown(step)
                            else:
                                st.markdown(step)

                st.write(message["content"])
                if message["role"] == "assistant" and "geopandas_link" in message:
                    # if "gdf" not in st.session_state:
                    # st.session_state.gdf = gpd.read_file(message["geopandas_link"])
                    gdf = gpd.read_file(
                        message["geopandas_link"]
                    )  # st.session_state.gdf
                    color_length = gdf["clust"].nunique()
                    print("color_length: ", color_length)
                    with st.expander("🗺️ **Show map**", expanded=True):
                        if color_length == 1:
                            # colormap = cm.linear.Set1_09.scale(0, 50).to_step(color_length)
                            map = gdf.explore(
                                column="clust",
                                tiles="CartoDB positron",
                                marker_kwds={"radius": 5},  # , cmap=colormap
                            )
                            folium_static(map, width=800)
                        else:
                            colormap = cm.linear.Set1_09.scale(
                                0, color_length + 1
                            ).to_step(color_length)
                            map = gdf[gdf["clust"] > -1].explore(
                                column="clust",
                                tiles="CartoDB positron",
                                marker_kwds={"radius": 5},
                                cmap=colormap,
                            )
                            folium_static(map, width=1000, height=500)
                st.caption(
                    f"⏰ {datetime.fromisoformat(message['timestamp']).strftime('%H:%M:%S')}"
                )

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.thread_manager.add_message(
                st.session_state.current_thread_id, "user", prompt
            )

            # structured input
            initial_state = {
                "messages": [{"role": "user", "content": prompt}],
                "location": None,
                "geocode_data": None,
                "error": None,
            }

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display bot response
            with st.chat_message("assistant"):
                # Create containers for thinking and response
                thinking_container = st.container()
                response_container = st.container()
                spatial_container = st.container()
                graph = st.session_state.chatbot.graph
                with st.spinner("🤔 Thinking..."):
                    with thinking_container:
                        with st.expander("🤔 **Thinking Process**", expanded=True):
                            thinking_placeholder = st.empty()
                            # thinking_steps, response = simple_chatbot_response(prompt)
                            displayed_steps = []
                            for chunk in graph.stream(
                                initial_state, stream_mode="updates"
                            ):
                                for key, value in chunk.items():
                                    # print(f"Node: {key}, Update: {value}")
                                    if "steps" in value.keys():
                                        steps = value.get("steps")  # thinking
                                        overpassInstructions = value.get(
                                            "overpassInstructions"
                                        )  # thinking
                                        displayed_steps += (
                                            ["🧠 Planning steps:\n"]
                                            + steps
                                            + [
                                                "\n💭 Creating overpass prompts for overpass:\n"
                                            ]
                                            + overpassInstructions
                                        )
                                        for displayed_step in displayed_steps:
                                            thinking_placeholder.markdown(
                                                "\n".join(overpassInstructions + steps)
                                            )

                                    if "overpassResponses" in value.keys():
                                        overpassResponses = value.get(
                                            "overpassResponses"
                                        )
                                        print("overpassResponses : ", overpassResponses)
                                        displayed_steps += (
                                            ["\n💭 Creating overpass queries:\n"]
                                            + overpassResponses
                                            + ["\n✅ Executing overpass queries\n"]
                                        )

                                        thinking_placeholder.markdown(
                                            "\n".join(
                                                ["\n💭 Creating overpass queries:\n"]
                                            )
                                        )
                                        for overpassResponse in overpassResponses:
                                            st.code(overpassResponse)
                                        thinking_placeholder.markdown(
                                            "\n".join(
                                                ["\n✅ Executing overpass queries\n"]
                                            )
                                        )
                                        # thinking = value.get("overpassResponses") #thinking, code

                                    if "geopandasData" in value.keys():
                                        geopandasData = value.get("geopandasData")

                                    # iterative
                                    if "stepCodes" in value.keys():
                                        stepCodes = value.get(
                                            "stepCodes"
                                        )  # thinking, code
                                        print("stepCodes : ")
                                        displayed_steps += (
                                            ["\n💭 Creating python codes:\n"]
                                            + [stepCodes]
                                            + ["\n✅ Executing python codes:\n"]
                                        )
                                        thinking_placeholder.markdown(
                                            "\n".join(["\n💭 Creating python codes:\n"])
                                        )

                                        st.code(stepCodes)
                                        thinking_placeholder.markdown(
                                            "\n".join(["\n✅ Executing python codes\n"])
                                        )

                                    if "finalResponse" in value.keys():
                                        response = value.get("finalResponse")  # rep
                            thinking_steps = displayed_steps

                            # Show final collapsed version
                        with st.expander(
                            "🤔 **Show Thinking Process**", expanded=False
                        ):
                            for step in thinking_steps:
                                if step.startswith("\n"):
                                    st.markdown(step)
                                elif step.startswith("overpass") or step.startswith(
                                    "python"
                                ):
                                    st.code(step)
                                else:
                                    st.markdown(step)

                # Show final response
                with response_container:
                    st.write(response.content)

                    # Add bot response to thread with thinking steps

                try:
                    with spatial_container:
                        gdf = gpd.read_file(geopandasData)  # st.session_state.gdf
                        color_length = gdf["clust"].nunique()
                        print(color_length)
                        with st.expander("🗺️ **Show map**", expanded=True):
                            if color_length == 1:
                                # colormap = cm.linear.Set1_09.scale(0, 50).to_step(color_length)
                                map = gdf.explore(
                                    column="clust",
                                    tiles="CartoDB positron",
                                    marker_kwds={"radius": 5},  # , cmap=colormap
                                )
                                folium_static(map, width=800)
                            else:
                                colormap = cm.linear.Set1_09.scale(
                                    0, color_length + 1
                                ).to_step(color_length)
                                map = gdf[gdf["clust"] > -1].explore(
                                    column="clust",
                                    tiles="CartoDB positron",
                                    marker_kwds={"radius": 5},
                                    cmap=colormap,
                                )
                                folium_static(map, width=1000, height=500)
                    st.session_state.thread_manager.add_message(
                        st.session_state.current_thread_id,
                        "assistant",
                        response.content,
                        thinking_steps,
                        geopandasData,
                    )
                except:
                    st.session_state.thread_manager.add_message(
                        st.session_state.current_thread_id,
                        "assistant",
                        response.content,
                        thinking_steps,
                    )

            st.rerun()

    # Footer
    st.divider()
    st.caption(
        "💡 **Tips:** Use the sidebar to manage your chat threads. Each conversation is automatically saved!"
    )


if __name__ == "__main__":
    main()
