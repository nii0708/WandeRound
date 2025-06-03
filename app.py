import pandas as pd
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Dict, List, Optional, TypedDict, Tuple, Any, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List
import uuid
from tools.osm import (
    get_osm_id_from_nominatim,
    process_overpass,
    get_exec_printed_result,
    get_response,
)
from tools.route import get_cluster, get_distance_matrix, get_route
from shapely.geometry import LineString
from dotenv import load_dotenv
import time


load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


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
    geopandasColumn: GeoColumn
    overpassInstructions: List[str]
    overpassResponses: List[str]
    overpassStatus: bool
    geopandasData: str
    location: Optional[str]
    geocode_data: Optional[Dict]
    error: Optional[str]
    trip: bool


class Extract(BaseModel):
    overpassInstructions: List[str] = Field(
        description="prompt instruction to get data via overpass API"
    )
    steps: List[str] = Field(description="steps to answer user question")
    location: str = Field(description="detailed location of user input")


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
    response = llm.invoke(location_extraction_prompt.format_messages(input=input_text))
    print("response : ", response.content)
    if response.content in ["True", "true", True, "TRUE"]:
        return {"trip": True}
    return {"trip": False}


def classifier(state):
    if state.get("trip", True):
        return "extract_location"
    else:
        return "usual"


def usual(state: AgentState) -> AgentState:
    human_message = state["messages"][-1]
    # print(human_message)
    final_answer = llm.invoke(human_message.content)
    return {"finalResponse": final_answer}


def extract_location(state: AgentState) -> AgentState:
    """Extract location from the user message"""

    human_message = state["messages"][-1]
    input_text = human_message.content

    location_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant that give plan to user trip by extracting location names from user messages, creating 2 text instructions to get data via overpass API based on user input, and at max three steps to plan the trip."
                "Identify the most specific location mentioned. Create 2 text instructions to get overpass API that return the body and Create at max 3 steps to plan the trip with the last step is to summarize the answers while the rest is to process data from overpass."
                "Return the name of the location, overpass instructions, and steps, nothing else.",
            ),
            ("human", "extract information from: {input}"),
        ]
    )

    # print('input_text (extract_location) : ',input_text)
    structured_llm = llm.with_structured_output(Extract)
    response = structured_llm.invoke(
        location_extraction_prompt.format_messages(input=input_text)
    )
    # print('response (extract_location) : ',response)
    return {
        "location": response.location,
        "steps": response.steps,
        "overpassInstructions": response.overpassInstructions,
        "stepsState": -int(len(response.steps)),
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
     Assistant is an expert OpenStreetMap Overpass API assistant.

        For each question that the user supplies, the assistant will reply with:
        The text of a valid Overpass API query using OSM geocode id that can be used to answer the question using geocode data: {geocode_data}. 
        The query should be enclosed by three backticks on new lines, denoting that it is a code block.
        the first line of the code is:
        [out:json];
        the last three lines of the code are:
            `// Output results
            out body;
            >;
            out skel qt;
            `
        if you need to filter based on proximity use this example:
        `// Filter tourism features that are within 50m of any canal
            (
            node.tourism_features(around.canals:50);
            way.tourism_features(around.canals:50);
            );`
        choose carefully if the code will use node, way, or relation or any combination of them:
        for example if user asked for canal only fetch `way` objects
    Assistant will reply with only a relevant Overpass API that return the body information in form of json.
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
            + [{"role": "assistant", "content": f"I encountered an error: {error}"}]
        }

    responses = []
    for instruction in overpassInstructions:
        # print('instruction : ',instruction)
        response = llm.invoke(
            response_prompt.format_messages(
                location=location,
                geocode_data=json.dumps(geocode_data, indent=2),
                human_input=instruction,
            )
        )
        responses.append(response.content)

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

    try:
        # code =  llm.invoke(code_prompt)
        print("params : \n", geopandas_link, column_list, featureValue, question)
        time.sleep(0.1)
        code = llm.invoke(
            code_creation_prompt.format_messages(
                geopandas_link=geopandas_link,
                column_list=", ".join(column_list),
                featureValue=", ".join(featureValue),
                question=question,
            )
        )
        print("PASS")
        print("code.content : \n", code)
        code_resp = get_exec_printed_result(process_overpass(code.content))
        return {
            "codeResult": code_resp,
            "stepCodes": code.content,
            "codeStatus": "PASS",
        }
    except Exception as e:
        print("err : ", e)
        print("code.content : \n", code)
        return {
            "error": e,
            "codeStatus": "ERROR",
            "stepCodes": [m for m in code if isinstance(m, BaseMessage)],
        }


def eval_code(state: AgentState) -> AgentState:
    evalState = state.get("evalState", 0)
    status = state.get("codeStatus", "PASS")
    step = state.get("stepsState", -3)
    if status == "PASS":
        step = step + 1
        return {"stepsState": step}
    elif status == "ERROR" and evalState < 3:
        # fix the step etc ....
        print("ERRR")
        err = state.get("err", "")
        curr_state = state.get("stepsState", None)
        step = state.get("steps", None)
        codes = state.get("stepCodes", "")
        question = step[curr_state]
        code = codes[-1]
        fix_prompt = f"""
        this question {question} create python code :
        {code} that return this {err} when executed. 
        
        Fix the code such that it will return the answer to the question
        """
        fix_code = llm.invoke(fix_prompt)
        evalState = evalState + 1
        print('state["stepCodes"] : ', state["stepCodes"])
        state["stepCodes"].insert(step, fix_code.content)
        return {"evalState": evalState}


def router(state):
    if state.get("evalState", 0) >= 3 or state.get("stepsState", -3) == -1:
        return "get_summary"
    else:
        return "get_code"


def get_summary(state: AgentState) -> AgentState:
    response = state.get("steps", None)
    results = state.get("codeResult", None)
    tour = state.get("route", None)

    response = response[-3:-1]
    summary = response[-1]
    wrap_up = "\n".join(
        [f"task : {resp} \n answer : {res}" for resp, res in zip(response, results)]
    )

    print("wrap_up : \n", wrap_up)

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
    final_answer = llm.invoke(
        summary_prompt.format_messages(summary=summary, tour=tour, wrap_up=wrap_up)
    )
    return {"finalResponse": final_answer}


class AgentGraph:
    def __init__(self):
        self.graph = self._create_graph()

    def _create_graph(self):
        # # Build the graph
        # workflow = StateGraph(AgentState)

        # # Add nodes
        # workflow.add_node("process_message", process_message)
        # workflow.add_node("usual", usual)

        # workflow.add_node("extract_location", extract_location)
        # workflow.add_node("geocode_location", geocode_location)
        # workflow.add_node("generate_response", generate_response)

        # workflow.add_node("execute_code", execute_code)
        # workflow.add_node("get_code", get_code)
        # workflow.add_node("eval_code", eval_code)
        # # workflow.add_node("router", router)
        # workflow.add_node("get_summary", get_summary)

        # # Add conditional edges
        # workflow.add_conditional_edges(
        #     "process_message",
        #     lambda state: "extract_location" if state.get("trip") else "usual",
        # )

        # workflow.add_conditional_edges(
        #     "extract_location",
        #     lambda state: "generate_response"
        #     if state.get("error")
        #     else "geocode_location",
        # )
        # workflow.add_edge("geocode_location", "generate_response")
        # workflow.add_edge("generate_response", "execute_code")
        # workflow.add_conditional_edges(
        #     "execute_code",
        #     lambda state: "get_code" if state.get("overpassStatus") else "usual",
        # )
        # # workflow.add_edge("execute_code", "get_code")
        # workflow.add_edge("get_code", "eval_code")
        # workflow.add_conditional_edges("eval_code", router)

        # workflow.add_edge("get_summary", END)
        # workflow.add_edge("usual", END)

        # # Set the entry point
        # workflow.set_entry_point("process_message")

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("process_message", process_message)
        workflow.add_node("usual", usual)

        workflow.add_node("extract_location", extract_location)
        workflow.add_node("geocode_location", geocode_location)
        workflow.add_node("generate_response", generate_response)

        workflow.add_node("execute_code", execute_code)
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
            lambda state: "get_code" if state.get("overpassStatus") else "usual",
        )
        # workflow.add_edge("execute_code", "get_code")
        workflow.add_edge("get_code", "eval_code")
        workflow.add_conditional_edges("eval_code", router)

        workflow.add_edge("get_summary", END)
        workflow.add_edge("usual", END)

        # Set the entry point
        workflow.set_entry_point("process_message")

        # Compile the graph
        return workflow.compile()
