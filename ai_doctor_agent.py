from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from tools import pubmed_tool
from fpdf import FPDF
from markdown2 import markdown

llm = init_chat_model(model='ollama:llama3.2:latest')
llm = llm.bind_tools([pubmed_tool])


def generate_pdf(report_text: str) -> bytes:
    html = markdown(report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.write_html(html)
    return bytes(pdf.output())


class State(TypedDict):
    messages: Annotated[list, add_messages]  # Storing messages for context: type and how to modify
    message_type: str | None
    name: str | None
    age: int | None
    gender: str | None
    history: str | None
    diagnosed: bool | None
    confirmed: bool | None
    report: str | None
    pdf_bytes: bytes | None


class SpecialistClassifier(BaseModel):
    message_type: Literal["cardiologist", "neurologist", "general practitioner"] = Field(
        ...,
        description="Classify which doctor is best suited for the patient's message"
    )


def classify_specialist(state: State):
    history = "\n".join(
        f"{msg.name}: {msg.content}"
        for msg in state["messages"][-8:]
    )

    system = """You are deciding which doctor should continue the conversation.  
        Possible outputs (one keyword only):  
        - cardiologist  
        - neurologist  
        - general practitioner  
        
        Examples:
        User: “Sharp chest pain when climbing stairs.”
        Output: cardiologist
        
        User: “Tingling in fingers, memory trouble.”
        Output: neurologist
        
        User: “Fever and fatigue for two days.”
        Output: general practitioner
        
        Now classify the following message:
        """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": history}
    ]

    response = llm.invoke(messages)

    choice = response.content.strip().lower()
    if choice not in ["cardiologist", "neurologist", "general practitioner"]:
        choice = "general practitioner"

    return {"message_type": choice}


def router(state: State):
    message_type = state.get('message_type', 'other')
    if message_type == 'cardiologist':
        return {"next": "cardiologist"}  # Temporary state until next node

    elif message_type == 'neurologist':
        return {"next": "neurologist"}

    return {"next": "general practitioner"}


def general_practitioner(state: State):
    patient_details = f"Patient Details:\nName: {state.get('name', 'Unknown')}\nAge: {state.get('age', 'Unknown')}" \
                      f"\nGender: {state.get('gender', 'Unknown')}\nHistory: {state.get('history', 'Unknown')}\n\n"

    system_content = """You are a caring and detail-oriented General Practitioner. First decide whether you truly 
    need PubMed based on your internal knowledge. If yes, explain why and call PubMed. If not, proceed with your 
    diagnosis without external search. Your role is to conduct initial consultations, collect comprehensive patient 
    information, and identify common illnesses. Ask follow-up questions to gather a full picture of the patient’s 
    condition, including lifestyle factors. If symptoms are outside your scope, refer the patient to the appropriate 
    specialist. Always prioritize patient comfort, clarity, and accuracy. Do not hallucinate any details, 
    and use only the information that the patient gives you.""" \
                     + patient_details

    gp_messages = [{"role": "system", "content": system_content}] + state["messages"]

    reply = llm.invoke(gp_messages)

    if reply.tool_calls:
        for tc in reply.tool_calls:
            if tc["name"] == pubmed_tool.name:
                tool_msg = pubmed_tool.invoke(tc["args"]["query"])
                # wrap into a tool message for conversation context

                from langchain_core.messages.tool import ToolMessage
                tm = ToolMessage(content=tool_msg, tool_call_id=tc["id"], name=pubmed_tool.name, query=tc["args"]["query"])
                state["messages"].append(tm)

            # Update conversation history
            gp_messages = [{"role": "system", "content": system_content}] + state["messages"]

            # Invoke LLM again—now grounded with PubMed results
            reply = llm.invoke(gp_messages)

    if "</think>" in reply.content and not reply.content.lstrip().startswith("<think>"):
        reply.content = "<think>" + reply.content

    new_message = {"role": "assistant", "content": reply.content, "name": "General Practitioner"}
    updated_messages = state.get("messages", []) + [new_message]
    return {"messages": updated_messages}


def cardiologist(state: State):
    cardio_messages = [
                          {"role": "system", "content": """You are a knowledgeable and experienced Cardiologist. You evaluate symptoms 
        related to the heart and circulatory system such as chest pain, palpitations, shortness of breath, 
        and dizziness. Ask targeted follow-up questions about cardiovascular history, medications, and risk factors.
        Provide a reasoned diagnosis or suggest relevant cardiac tests when needed."""}
                      ] + state["messages"]

    reply = llm.invoke(cardio_messages)

    if reply.tool_calls:
        for tc in reply.tool_calls:
            if tc["name"] == pubmed_tool.name:
                tool_msg = pubmed_tool.invoke(tc["args"]["query"])
                # wrap into a tool message for conversation context

                from langchain_core.messages.tool import ToolMessage
                tm = ToolMessage(content=tool_msg, tool_call_id=tc["id"], name=pubmed_tool.name)
                state["messages"].append(tm)

            # Update conversation history
            gp_messages = [{"role": "system", "content": cardio_messages}] + state["messages"]

            # Invoke LLM again—now grounded with PubMed results
            reply = llm.invoke(gp_messages)

    if "</think>" in reply.content and not reply.content.lstrip().startswith("<think>"):
        reply.content = "<think>" + reply.content

    new_message = {"role": "assistant", "content": reply.content, "name": "Cardiologist"}
    updated_messages = state.get("messages", []) + [new_message]
    return {"messages": updated_messages}


def neurologist(state: State):
    neuro_messages = [
                         {"role": "system", "content": """You are a specialized and analytical Neurologist. Focus on symptoms related 
        to the nervous system, including headaches, dizziness, numbness, memory loss, and coordination issues. Ask 
        relevant neurological assessment questions to narrow down possible conditions. Consider medical history and 
        recent symptom patterns to form a working diagnosis or recommend neurological evaluation."""}
                     ] + state["messages"]

    reply = llm.invoke(neuro_messages)

    if reply.tool_calls:
        for tc in reply.tool_calls:
            if tc["name"] == pubmed_tool.name:
                tool_msg = pubmed_tool.invoke(tc["args"]["query"])
                # wrap into a tool message for conversation context

                from langchain_core.messages.tool import ToolMessage
                tm = ToolMessage(content=tool_msg, tool_call_id=tc["id"], name=pubmed_tool.name)
                state["messages"].append(tm)

            # Update conversation history
            gp_messages = [{"role": "system", "content": neuro_messages}] + state["messages"]

            # Invoke LLM again—now grounded with PubMed results
            reply = llm.invoke(gp_messages)

    if "</think>" in reply.content and not reply.content.lstrip().startswith("<think>"):
        reply.content = "<think>" + reply.content

    new_message = {"role": "assistant", "content": reply.content, "name": "Neurologist"}
    updated_messages = state.get("messages", []) + [new_message]
    return {"messages": updated_messages}


def check_diagnosis(state: State) -> dict:
    msg = state["messages"][-1].content.lower()
    diagnosis_messages = [
        {"role": "system", "content": """Your goal is to determine if the doctor has reached a conclusive diagnosis.
            You must only reply using the following words:
            - true if a conclusive diagnosis has been reached,
            - false if a conclusive diagnosis has not been reached.
            A conclusive diagnosis is where a treatment/medicine is recommended, or where the patient is 
            referred to for a future visit."""},

        {"role": "user", "content": msg}
    ]

    response = llm.invoke(diagnosis_messages)
    diagnosed = "true" in response.content

    return {"diagnosed": diagnosed}


def reporter(state: State):
    reporter_messages = [
        {"role": "system", "content": f""""You are a professional medical report author. Using only the patient data 
        and final diagnosis previously gathered, create a clear, precise medical report with the following sections:
            
            ---  
            **Patient Information**  
            - Name  
            - Age
            - Gender  
            - Medical History (brief summary)  
            
            **Chief Complaint & Presenting Symptoms**  
            
            **Physical Findings & Diagnostic Observations** (as described by doctor)  
            
            **Final Diagnosis** (as determined by the physician agent)  
            
            **Assessment & Rationale** (brief reasoning behind the diagnosis)  
            
            **Treatment Recommendations or Next Steps** (medications, referrals, follow‑up suggestions)  
            
            End with a short courteous note:  
            “Thank you for using our medical assistant service. Wishing you good health."
            
            Also, add this disclaimer in bold: "This is not an official report. The recommendations in this report 
            are merely suggestions and must be taken with caution." --- Give only the report and don't add any 
            additional text. """},

        {"role": "user", "content": f"""Here is the conversation from which you can summarize: {state}"""}
    ]

    report = llm.invoke(reporter_messages)

    return {"messages": state["messages"], "diagnosed": state["diagnosed"], "report": report.content}


graph_builder = StateGraph(State)

# graph_builder.add_node("restart", restart)
graph_builder.add_node("classify_specialist", classify_specialist)
graph_builder.add_node("router", router)
graph_builder.add_node("general_practitioner", general_practitioner)
graph_builder.add_node("cardiologist", cardiologist)
graph_builder.add_node("neurologist", neurologist)
graph_builder.add_node("check_diagnosis", check_diagnosis)
graph_builder.add_node("reporter", reporter)

graph_builder.add_edge(START, "classify_specialist")
graph_builder.add_edge("classify_specialist", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get('next'),
    {
        "cardiologist": "cardiologist",
        "neurologist": "neurologist",
        "general practitioner": "general_practitioner"
    }
)

graph_builder.add_edge("general_practitioner", "check_diagnosis")
graph_builder.add_edge("cardiologist", "check_diagnosis")
graph_builder.add_edge("neurologist", "check_diagnosis")

graph_builder.add_conditional_edges(
    "check_diagnosis",
    lambda s: "reporter" if s.get("diagnosed") else END,
    {"reporter": "reporter", END: END}
)

graph_builder.add_edge("reporter", END)

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)


def run_chatbot():
    print("Hello! I'm your virtual medical assistant. Let me collect a few basic details to get started.")

    name = input("Name: ")
    age = int(input("Age: "))
    gender = input("Gender: ")
    history = input("Previous medical history: ")

    state = {
        "messages": [],
        "message_type": None,
        "name": name,
        "age": age,
        "gender": gender,
        "history": history,
        "diagnosed": False
    }

    config = {"configurable": {"thread_id": "2"}}

    while True:

        # Prompt for user input after any assistant message
        user_input = input("\nYou: ")
        if user_input.lower() == 'q':
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state, config)

        print(f"{state['messages'][-1].name}: {state['messages'][-1].content}")

        if state["diagnosed"]:
            print("Final report:")
            print(state["report"])
            break


if __name__ == '__main__':
    run_chatbot()
