import pandas as pd
import getpass
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Dict, List, Optional, TypedDict, Tuple, Any, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field
from typing import List
import uuid
from tools.osm import (
    get_osm_id_from_nominatim,
    process_overpass,
    get_exec_printed_result,
    get_response,
    reverse_geocode,
)
from tools.route import get_cluster, get_distance_matrix, get_route
from shapely.geometry import LineString
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
if "GOOGLE_MODEL" not in os.environ:
    os.environ["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL", "")
if "OPENAI_MODEL" not in os.environ:
    os.environ["OPENAI_MODEL"] = os.getenv("OPENAI_MODEL", "")
if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
if "DEEPSEEK_MODEL" not in os.environ:
    os.environ["DEEPSEEK_MODEL"] = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class GeoColumn(TypedDict):
    column: List[str]
    featureValue: List[str]


# Type definitions for the state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    finalResponse: Annotated[list[BaseMessage], add_messages]
    summary: str
    route: List[int]
    steps: List[str]
    stepsState: int
    stepCodes: Annotated[list[BaseMessage], add_messages]
    codeResult: Annotated[list[BaseMessage], add_messages]
    codeStatus: str
    evalState: int
    evalCode: str
    geopandasColumn: GeoColumn
    overpassInstructions: List[str]
    overpassResponses: List[str]
    overpassStatus: bool
    geopandasData: str
    mapLabels: Optional[Dict[str, Any]]
    location: Optional[str]
    geocode_data: Optional[Dict]
    error: Optional[str]
    trip: bool


class Extract(BaseModel):
    overpassInstructions: List[str] = Field(
        description="prompt instructions to get data via overpass API"
    )
    steps: List[str] = Field(description="steps to answer user question")
    location: str = Field(description="detailed location of user input")


class AgentGraph:
    def __init__(self):
        self.graph = self._create_graph()
        self.llm = None

    def model_choose(self, model_source):
        if model_source == "GOOGLE":
            self.llm = ChatGoogleGenerativeAI(
                model=os.environ["GOOGLE_MODEL"],
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif model_source == "OPENAI":
            self.llm = ChatOpenAI(
                model=os.environ["OPENAI_MODEL"],
                temperature=0,
                max_tokens=None,
                max_retries=2,
                streaming=True,
            )
        elif model_source == "DEEPSEEK":
            self.llm = ChatOpenAI(
                model=os.environ["DEEPSEEK_MODEL"],
                temperature=0,
                max_tokens=None,
                max_retries=2,
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
                streaming=True,
            )

    def _create_graph(self):
        def process_message(state: AgentState) -> AgentState:
            human_message = state["messages"][-1]
            input_text = human_message.content

            location_extraction_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a trip/vacation planning classifier. Your task is to analyze the user's input "
                        "and determine if they are asking you to plan a trip, itinerary, or travel arrangements."
                        "\n\n"
                        "If the input explicitly or implicitly requests assistance with planning travel "
                        "(e.g., Create a trip plan to Bangkok, plan a trip to Bali "
                        "detailing activities for a journey), return `true`."
                        "\n"
                        "Otherwise, if the input is about a different topic, return `false`."
                        "\n\n"
                        "Respond with only `true` or `false`.",
                    ),
                    ("human", "{input}"),
                ]
            )
            response = self.llm.invoke(
                location_extraction_prompt.format_messages(input=input_text)
            )
            if response.content in ["True", "true", True, "TRUE"]:
                return {"trip": True}
            return {"trip": False}

        def classifier(state):
            if state.get("trip", True):
                return "extract_location"
            else:
                return "usual"

        def usual(state: AgentState) -> AgentState:
            history = state["messages"]
            system_msg = SystemMessage(
                content=(
                    "You are WandeRound, a friendly travel-planning assistant. "
                    "Use the full conversation so far to give context-aware, "
                    "natural follow-up answers. Reply in the same language as "
                    "the user's most recent message."
                )
            )
            final_answer = self.llm.invoke([system_msg, *history])
            return {"finalResponse": final_answer}

        def extract_location(state: AgentState) -> AgentState:
            """Extract location from the user message"""

            human_message = state["messages"][-1]
            input_text = human_message.content

            location_extraction_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a travel planning assistant. Extract trip information from the user message.\n\n"
                        "Return ONLY a valid JSON object with exactly these fields:\n"
                        "{{\n"
                        '  "location": "the most specific location name mentioned",\n'
                        '  "overpassInstructions": ["instruction 1 for overpass query", "instruction 2 for overpass query"],\n'
                        '  "steps": ["step 1", "step 2", "step 3 (summarize)"]\n'
                        "}}\n\n"
                        "Rules:\n"
                        "- overpassInstructions: exactly 2 instructions describing what OSM data to fetch\n"
                        "- steps: at most 3 steps, the last step must be to summarize the answers\n"
                        "- Do not fetch buildings\n"
                        "- Return ONLY the JSON object, no markdown, no explanation",
                    ),
                    ("human", "extract information from: {input}"),
                ]
            )

            response = self.llm.invoke(
                location_extraction_prompt.format_messages(input=input_text)
            )

            content = response.content.strip()
            # Strip thinking blocks emitted by reasoning models
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Strip markdown code fences
            content = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
            # Extract JSON object (from first { to last })
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError(f"No JSON found in model response: {content[:300]}")
            data = json.loads(content[start:end])

            return {
                "location": data["location"],
                "steps": data["steps"],
                "overpassInstructions": data["overpassInstructions"],
                "stepsState": -int(len(data["steps"])),
                "evalState": 0,
            }

        # Define the geocoding node
        def geocode_location(state: AgentState) -> AgentState:
            """Get geocode data for the extracted location"""
            location = state.get("location")

            if not location:
                return {"error": "No location found in the message."}

            geocode_data, error = get_osm_id_from_nominatim(location)

            if error:
                return {"error": error}

            return {"geocode_data": geocode_data}

        def generate_response(state: AgentState) -> AgentState:
            """Generate a response based on the location and geocode data"""
            location = state.get("location", "Unknown location")
            geocode_data = state.get("geocode_data", {})
            error = state.get("error")
            overpassInstructions = state.get("overpassInstructions", [])

            response_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
            You are an expert OpenStreetMap Overpass API assistant.

            The location has been geocoded to: {geocode_data}
            Use the lat/lon coordinates with `around:` radius queries. Do NOT use area() queries.

            For each question, reply with a valid Overpass API query enclosed in triple backticks.
            Always use this structure:
            ```
            [out:json];
            (
              node["KEY"~"VALUE"](around:3000,LAT,LON);
              way["KEY"~"VALUE"](around:3000,LAT,LON);
            );
            out body;
            >;
            out skel qt;
            ```

            Replace LAT and LON with the actual coordinates from the geocode data.
            Use around:3000 (3km radius) by default.
            Choose node, way, or both depending on what is being searched.
            Do NOT fetch buildings.
            Reply with ONLY the Overpass query code block, nothing else.
                """,
                    ),
                    (
                        "human",
                        "Please, create the overpas query for: {human_input} with this geocode data: {geocode_data}",
                    ),
                ]
            )

            if error:
                return {
                    "messages": state["messages"]
                    + [
                        {
                            "role": "assistant",
                            "content": f"I encountered an error: {error}",
                        }
                    ]
                }

            batch_inputs = [
                response_prompt.format_messages(
                    location=location,
                    geocode_data=json.dumps(geocode_data, indent=2),
                    human_input=instruction,
                )
                for instruction in overpassInstructions
            ]
            batch_results = self.llm.batch(batch_inputs)
            responses = [r.content for r in batch_results]

            return {"overpassResponses": responses}

        def execute_code(state: AgentState) -> AgentState:
            """Generate a response based on the location and geocode data"""
            try:
                overpassResponses = state.get("overpassResponses", None)

                combined_gdf = get_response(overpassResponses)

                combined_gdf = combined_gdf[~combined_gdf["geometry"].is_empty]

                cluster_gdf = get_cluster(combined_gdf)

                clust_centroid, distance_matrix = get_distance_matrix(cluster_gdf)

                route = get_route(distance_matrix)

                os.makedirs("./data", exist_ok=True)
                file_name = f"./data/sample_{uuid.uuid4()}.gpkg"

                combined_gdf.to_file(file_name)

                df_order = pd.DataFrame(
                    {"clust": route, "order": [i for i in range(len(route))]}
                )

                df_order = clust_centroid.merge(df_order, on="clust")

                path = df_order.sort_values("order")[0].to_list()

                path = LineString(path + [path[0]])

                return {
                    "overpassStatus": True,
                    "geopandasData": file_name,
                    "route": route,
                    "geopandasColumn": {
                        "column": combined_gdf.columns.to_list(),
                        "featureValue": list(combined_gdf["feature"].unique()),
                    },
                }
            except:
                return {"overpassStatus": False}

        def overpass_status(state):
            if state.get("overpassStatus", True):
                return "get_code"
            else:
                return "usual"

        def generate_map_labels(state: AgentState) -> AgentState:
            """Build cluster labels from reverse-geocoded area names, plus UI strings via LLM."""
            file = state.get("geopandasData")
            if not file:
                return {"mapLabels": None}
            try:
                import geopandas as gpd

                gdf = gpd.read_file(file)

                # Compute centroid per cluster (skip -1 = noise/unclustered)
                cluster_ids = sorted({int(c) for c in gdf["clust"].dropna().unique()})
                cluster_centroids: Dict[int, Tuple[float, float]] = {}
                for cid in cluster_ids:
                    if cid < 0:
                        continue
                    group = gdf[gdf["clust"] == cid]
                    if group.empty:
                        continue
                    centroids = group.geometry.centroid
                    lat = float(centroids.y.mean())
                    lon = float(centroids.x.mean())
                    cluster_centroids[cid] = (lat, lon)

                # Reverse-geocode each cluster centroid. Nominatim asks for ≤1 req/sec.
                area_names: Dict[int, str] = {}
                for cid, (lat, lon) in cluster_centroids.items():
                    name = reverse_geocode(lat, lon)
                    if name:
                        area_names[cid] = name
                    time.sleep(1.1)

                # Deduplicate identical area names by suffixing (e.g. "Malioboro 2")
                seen_counts: Dict[str, int] = {}
                cluster_labels: Dict[str, str] = {}
                for cid in cluster_ids:
                    if cid < 0:
                        continue  # filled in by UI string "unclustered"
                    raw = area_names.get(cid)
                    if not raw:
                        cluster_labels[str(cid)] = f"Area {cid}"
                        continue
                    seen_counts[raw] = seen_counts.get(raw, 0) + 1
                    cluster_labels[str(cid)] = (
                        raw if seen_counts[raw] == 1 else f"{raw} {seen_counts[raw]}"
                    )

                human_message = state["messages"][-1]
                user_text = (
                    human_message.content
                    if hasattr(human_message, "content")
                    else str(human_message)
                )

                # LLM only translates the UI strings to the user's language.
                ui_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You generate short UI strings for an interactive map of points of interest.\n"
                            "RESPOND IN THE SAME LANGUAGE AS THE USER MESSAGE.\n"
                            "Return ONLY a valid JSON object — no markdown, no commentary. Shape:\n"
                            "{{\n"
                            '  "name": "label for the Name field",\n'
                            '  "type": "label for the Type/Category field",\n'
                            '  "address": "label for the Address field",\n'
                            '  "cluster": "label for the Area field",\n'
                            '  "unclustered": "label used for points that do not belong to any area",\n'
                            '  "legendTitle": "short title for the legend, max 2 words",\n'
                            '  "noName": "placeholder when a place has no name"\n'
                            "}}\n"
                            "Keep every label short and human-friendly.",
                        ),
                        (
                            "human",
                            "User message: {user_text}\n\nGenerate the JSON now.",
                        ),
                    ]
                )

                ui_labels: Dict[str, str] = {}
                try:
                    response = self.llm.invoke(
                        ui_prompt.format_messages(user_text=user_text)
                    )
                    content = response.content.strip()
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    content = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start != -1 and end > start:
                        ui_labels = json.loads(content[start:end])
                except Exception as e:
                    print(f"generate_map_labels UI strings error: {e}")

                labels = {"ui": ui_labels, "clusters": cluster_labels}
                return {"mapLabels": labels}
            except Exception as e:
                print(f"generate_map_labels error: {e}")
                return {"mapLabels": None}

        def get_code(state: AgentState) -> AgentState:
            # (geopandas_link,column,question):
            geopandas_link = state.get("geopandasData", None)
            column = state.get("geopandasColumn", None)
            curr_state = state.get("stepsState", None)
            step = state.get("steps", None)
            question = step[curr_state]

            column_list = column.get("column")
            featureValue = column.get("featureValue")

            code_creation_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
            You are an expert in Geospatial Data Science, specifically with Geopandas library.
            Your task is to generate Geopandas code to answer a user's question, which will be executed within a `PythonAstREPLTool`.

            Here's the context you'll be working with:

            1.  **Geopandas Data:** You will be provided with a link to a Geopandas GeoPackage (`.gpkg`) file.
                * `geopandas_link`: {geopandas_link}
                * `geopandas columns` : {column_list}
                * `geopandas feature values` : {featureValue}

            2.  **Geopandas DataFrame Structure:** The Geopandas DataFrame loaded from the provided link will have at least the following columns, along with their descriptions and potential values:

                * `id`:
                    * **Description:** Unique identifier for the geospatial object. 
                * `geometry`:
                    * **Description:** The geometric representation of the object (e.g., Point, LineString, Polygon). This is the standard Geopandas 'geometry' column.
                * `name`:
                    * **Description:** The common name of the object.
                * `feature`:
                    * **Description:** A general descriptive tag for the object's distinct characteristic or primary function, if not covered by other specific tags. 
                    * **Examples:** {featureValue}
                * `clust`:
                    * **Description:** cluster of geospatial object
                    * **Examples:** 0, 1, 2

            3.  **User's Question:**
                * `question`: {question}

            **Your Goal:**

            Generate clean, efficient, and correct Geopandas Python code that, when executed in a `PythonAstREPLTool`, will provide the answer to the `question`.
            If possible, always omit using 'addr:city' to filter out location is this column is often incomplete. 
            And always refer geopandas feature columns values to filter value in feature columns.
            

            **Important Considerations for Code Generation:**

            * **Load Data:** Always start by loading the Geopandas DataFrame using `gpd.read_file(geopandas_link)`. Assign it to a variable named `gdf`.
            * **Column Names:** Strictly adhere to the provided column names (e.g., `feature`, `building:use`).
            * **Output:** The final output of your code should be the direct answer to the question that is printed. This is a specific value, a count, a list, etc. Do not output whole dataframe. 
            * **Error Handling:** Assume the input `.gpkg` file is valid and contains the described columns. Do not add explicit error handling unless specifically requested.
            * **Clarity and Conciseness:** Write Pythonic and readable code. Prioritize to do the operation on cluster level. if the user question features are not in `geopandas feature value`, use similar `geopandas feature value` that match the question.
            * **Avoid Unnecessary Imports:** Only import `geopandas` as `gpd` and other necessary libraries if truly required (e.g., `numpy` for specific operations,).
            """,
                    ),
                    ("human", "create code to answer: {question}"),
                ]
            )
            # print("params : \n", geopandas_link, column_list, featureValue, question)
            try:
                time.sleep(0.1)
                code = self.llm.invoke(
                    code_creation_prompt.format_messages(
                        geopandas_link=geopandas_link,
                        column_list=", ".join(column_list),
                        featureValue=", ".join(featureValue),
                        question=question,
                    )
                )
                # code = None
                # print("get_code : ", code.content)
                # code_resp = get_exec_printed_result(process_overpass(code.content))
                return {
                    # "codeResult": code_resp,
                    "stepCodes": code.content,
                    "codeStatus": "PASS",
                }
            except Exception as e:
                print("get_code ERR : \n", e)
                # if code is None:
                return {
                    "error": "No code generated",
                    "codeStatus": "ERROR",
                    "stepCodes": "NO CODE GENERATED",
                }
            # else:

        def execute_python(state: AgentState) -> AgentState:
            evalState = state.get("evalState", 0)
            step = state.get("stepsState", -3)
            try:
                code = state.get("stepCodes")[-1]
                code_resp = get_exec_printed_result(process_overpass(code))
                return {
                    "codeResult": code_resp,
                    "codeStatus": "PASS",
                    "stepsState": step + 1,
                }
            except Exception as e:
                return {
                    "error": e,
                    "codeStatus": "ERROR",
                    "stepCodes": [m for m in code if isinstance(m, BaseMessage)],
                }

        def eval_code(state: AgentState) -> AgentState:
            evalState = state.get("evalState", 0)
            stepsState = state.get("stepsState", -3)

            status = state.get("codeStatus", "PASS")

            if status == "PASS":
                code = state.get("stepCodes")[-1]
                state["stepCodes"][-1] = code.content
                return {"stepsState": stepsState}
            elif status == "ERROR" and evalState < 3:
                err = state.get("err", "")
                geopandas_link = state.get("geopandasData", None)

                curr_state = state.get("stepsState", None)
                step = state.get("steps", [])
                codes = state.get("stepCodes", [])
                question = step[curr_state]
                code = codes[-1]
                fix_prompt_template = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                        You are an expert at geopandas and want to fix the python code : {code}  
                        that return this {err} when executed.
                        to answer this question: {question}  
                        
                        
                        Fix the python code such that it will return the code to answer to the question
                        Follow this rules:
                        * Do not make any comments in the code.
                        * Make the code as concise as possible.
                        * use already existed geopandas data: {geopandas_link}
                        """,
                        ),
                        ("human", "fix the {code} code to answer: {question}"),
                    ]
                )
                # fix_prompt = f"""
                # this question: {question}
                # create python code : {code}
                # that return this {err} when executed.

                # Fix the python code such that it will return the code to answer to the question
                # """
                fix_code = self.llm.invoke(
                    fix_prompt_template.format_messages(
                        geopandas_link=geopandas_link,
                        question=question,
                        code=code,
                        err=err,
                    )
                )
                evalState = evalState + 1
                state["stepCodes"][-1] = fix_code.content
                # print('state["stepCodes"] : \n', state["stepCodes"])
                return {"evalState": evalState}

        def router(state):
            status = state.get("codeStatus", "PASS")
            if state.get("evalState", 0) >= 3:
                return "usual"
            elif state.get("stepsState", -3) == -1:
                return "get_summary"
            elif status == "ERROR":
                return "eval_code"
            else:
                return "get_code"

        def get_summary(state: AgentState) -> AgentState:
            response = state.get("steps", None)
            results = state.get("codeResult", None)
            tour = state.get("route", None)

            response = response[-3:-1]
            summary = response[-1]
            wrap_up = "\n".join(
                [
                    f"task : {resp} \n answer : {res}"
                    for resp, res in zip(response, results)
                ]
            )

            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert tour guide who provides personalized and efficient tour suggestions. 
                        Your recommendations are always data-driven, utilizing information about the shortest routes between clusters of attractions and task-answer pairs to create optimal experiences.
                        
                        When a user asks for a tour suggestion, you will:
                        
                        Analyze the "shortest route between clusters" data to understand the most efficient paths connecting different groups of attractions.
                        Consult the "task answer pairs" to understand specific user preferences, common questions, or desired outcomes related to tour activities.
                        
                        Synthesize this information to generate a tour suggestion that is:
                        Efficient: Minimizing travel time and maximizing sightseeing.
                        Relevant: Tailored to the user's likely interests based on common patterns or previous interactions (if available).
                        Informative: Providing key details about each stop and why it's recommended.
                        And Friendly.
                        """,
                    ),
                    (
                        "human",
                        """
                        Based on the provided data, craft a comprehensive tour suggestion for the user to answer this questions: {summary}.

                        on this data

                        1. shortest route between clusters 
                        {tour}
                        2. task answer pairs;
                        {wrap_up}
                    """,
                    ),
                ]
            )
            final_answer = self.llm.invoke(
                summary_prompt.format_messages(
                    summary=summary, tour=tour, wrap_up=wrap_up
                )
            )
            return {"finalResponse": final_answer}

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("process_message", process_message)
        workflow.add_node("usual", usual)

        workflow.add_node("extract_location", extract_location)
        workflow.add_node("geocode_location", geocode_location)
        workflow.add_node("generate_response", generate_response)

        workflow.add_node("execute_code", execute_code)
        workflow.add_node("generate_map_labels", generate_map_labels)
        workflow.add_node("execute_python", execute_python)
        workflow.add_node("get_code", get_code)
        workflow.add_node("eval_code", eval_code)
        # workflow.add_node("router", router)
        workflow.add_node("get_summary", get_summary)

        # Add conditional edges
        workflow.add_conditional_edges(
            "process_message",
            lambda state: "extract_location" if state.get("trip") else "usual",
        )
        workflow.add_edge("extract_location", "geocode_location")

        workflow.add_edge("geocode_location", "generate_response")
        workflow.add_edge("generate_response", "execute_code")
        workflow.add_conditional_edges(
            "execute_code",
            lambda state: "generate_map_labels" if state.get("overpassStatus") else "usual",
        )
        workflow.add_edge("generate_map_labels", "get_code")

        workflow.add_edge("get_code", "eval_code")
        workflow.add_edge("eval_code", "execute_python")
        workflow.add_conditional_edges("execute_python", router)

        workflow.add_edge("get_summary", END)
        workflow.add_edge("usual", END)

        # Set the entry point
        workflow.set_entry_point("process_message")

        # Compile the graph
        return workflow.compile()
