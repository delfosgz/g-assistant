import os
from datetime import datetime
import operator
from numpy import rint
from typing_extensions import Annotated, List, Literal, Optional, Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage,
    AIMessage,
    ToolMessage,
    merge_message_runs,
    BaseMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from zoneinfo import ZoneInfo 
import google.auth
from googleapiclient.discovery import build
from pydantic import BaseModel, Field, model_validator
import uuid
from trustcall import create_extractor
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
import configuration 

# ==================== CONFIG ====================
load_dotenv()
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0)

# ==================== SCHEMAS ====================
class UserProfile(BaseModel):
    """Perfil completo de usuario"""
    user_name: Optional[str] = Field(None, description='Nombre del usuario')
    user_location: Optional[str] = Field(None, description='Ubicación geográfica')
    interests: List[str] = Field(default_factory=list, description="Hobbies y gustos")
    timezone: str = Field(default="America/Mexico_City", description="Zona horaria")

class FormatGetCalendarEvents(BaseModel):
    """Input para buscar eventos."""
    start_date: str = Field(
        description="Fecha de inicio en 'YYYY-MM-DD'",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    end_date: Optional[str] = Field(
        None,
        description="Fecha de fin 'YYYY-MM-DD' (opcional)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    
    @model_validator(mode='after')
    def check_date_order(self):
        if self.end_date is None:
            return self
        fmt = "%Y-%m-%d"
        start = datetime.strptime(self.start_date, fmt)
        end = datetime.strptime(self.end_date, fmt)
        if end <=   start:
            raise ValueError(f"End date must be after start date")
        return self

class FormatCreateEvent(BaseModel):
    """Esquema para crear eventos."""
    summary: str = Field(
        description="Título del evento"
    )
    start_datetime: str = Field(
        description="Fecha inicio en ISO: 'YYYY-MM-DDTHH:MM:SS'",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
    )
    end_datetime: str = Field(
        description="Fecha fin en ISO: 'YYYY-MM-DDTHH:MM:SS'",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
    )
    description: Optional[str] = Field(None, description="Detalles adicionales")
    recurrence: Optional[str] = Field(
        None,
        description="RRULE en formato iCalendar (ej: RRULE:FREQ=WEEKLY;BYDAY=MO,FR)",
        pattern=r"^RRULE:FREQ=(DAILY|WEEKLY|MONTHLY|YEARLY)(;.*)?$"
    )
    
    @model_validator(mode='after')
    def check_date_order(self):
        fmt = "%Y-%m-%dT%H:%M:%S"
        start = datetime.strptime(self.start_datetime, fmt)
        end = datetime.strptime(self.end_datetime, fmt)
        if end <= start:
            raise ValueError("End datetime must be after start datetime")
        return self

# ==================== STATE ====================
class AgentState(MessagesState):
    """Estado extendido del agente"""
    # Flag para indicar si se debe actualizar memoria
    #should_update_memory: bool = False
    # Acumulador de información para memoria
    #memory_updates: List[str] = []

# ==================== LAYER 1: TOOLS ====================

def get_calendar_service():
    """Autentica y retorna el servicio de Calendar API."""
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds, _ = google.auth.default(scopes=SCOPES)
    return build('calendar', 'v3', credentials=creds)

@tool(args_schema=FormatGetCalendarEvents)
def getCalendarEvents(start_date: str, end_date: str = None) -> str:
    """Obtiene eventos del calendario para una fecha o rango."""
    service = get_calendar_service()
    zona_cdmx = ZoneInfo("America/Mexico_City")
    
    dt_start = datetime.strptime(start_date, "%Y-%m-%d").replace(
        hour=0, minute=0, second=0, tzinfo=zona_cdmx
    )
    
    if end_date:
        dt_end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=zona_cdmx
        )
    else:
        dt_end = dt_start.replace(hour=23, minute=59, second=59)

    try:
        events_result = service.events().list(
            calendarId='delfosgz@gmail.com',
            timeMin=dt_start.isoformat(), 
            timeMax=dt_end.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        
        if not events:
            return f"No hay eventos entre {dt_start.date()} y {dt_end.date()}."
        
        result = f"Eventos ({dt_start.date()} a {dt_end.date()}):\n"
        for event in events:
            raw_start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'Sin título')
            result += f"- {raw_start}: {summary}\n"
        return result
    
    except Exception as e:
        return f"Error al obtener eventos: {str(e)}"

@tool(args_schema=FormatCreateEvent)
def createCalendarEvent(
    summary: str, 
    start_datetime: str, 
    end_datetime: str, 
    description: str = None, 
    recurrence: str = None
) -> str:
    """Crea un evento en Google Calendar."""
    service = get_calendar_service()
    
    event_body = {
        'summary': summary,
        'start': {'dateTime': start_datetime, 'timeZone': 'America/Mexico_City'},
        'end': {'dateTime': end_datetime, 'timeZone': 'America/Mexico_City'},
    }
    if description:
        event_body['description'] = description
    if recurrence:
        event_body['recurrence'] = [recurrence]
    
    try:
        event = service.events().insert(
            calendarId='delfosgz@gmail.com',
            body=event_body
        ).execute()
        return f"Evento creado: {event.get('htmlLink')}"
    except Exception as e:
        return f"Error al crear evento: {str(e)}"

# Lista de herramientas de calendario
CALENDAR_TOOLS = [getCalendarEvents, createCalendarEvent]
CALENDAR_TOOL_NAMES = {tool.name for tool in CALENDAR_TOOLS}

# ==================== LAYER 2: MEMORY ====================

class ExtractionDecision(BaseModel):
    """Decide si extraer información personal nueva o no."""
    should_extract: bool = Field(
        description="True si HAY información personal nueva que DEBE extraerse. False de lo contrario."
)

class MemoryManager:
    """Gestiona la persistencia del perfil de usuario"""
    
    def __init__(self, llm):
        # trust call extractor
        self.extractor = create_extractor(
            llm,
            tools=[UserProfile],
            tool_choice='UserProfile'
        )
        # classifier LLM
        self.classifier_llm = ChatOpenAI(
            model="gpt-5-nano-2025-08-07",
            temperature=0,
            max_tokens=50
        )
        # Heuristics keywords
        self.negative_keywords = {
            "qué", "cuándo", "dónde", "cómo", "quién", "cuál", "cuáles"
        }
        self.positive_patterns = [
            "me llamo", "mi nombre es", "soy de", "vivo en",
            "trabajo como", "me gusta", "me encanta", "mi hobby es",
            "soy un", "soy una", "me dedico a"
        ]
        
    
    def get_namespace(self, config: RunnableConfig) -> tuple:
        """Obtiene el namespace del usuario actual"""
        configurable = configuration.Configuration.from_runnable_config(config)
        return ('user_profile', configurable.user_id)
    
    def load_profile(self, store: BaseStore, config: RunnableConfig) -> Optional[dict]:
        """Carga el perfil del usuario desde el store"""
        namespace = self.get_namespace(config)
        items = store.search(namespace)
        return items[0].value if items else None
    
    def _llm_should_extract(self, message_content: str) -> bool:
        """
        Usa LLM pequeño para decisión precisa.
        """
        
        system_prompt = """
            Determina si el mensaje contiene información PERSONAL NUEVA del USUARIO que debería guardarse.

            INFORMACIÓN PERSONAL VÁLIDA (extraer):
            - Nombre del usuario (no de otras personas)
            - Ubicación/residencia del usuario
            - Gustos/hobbies/intereses del usuario
            - Rutinas/hábitos del usuario
            - Preferencias del usuario
            - Profesión/ocupación del usuario

            NO EXTRAER:
            - Nombres de otras personas ("tengo una amiga llamada Ana")
            - Preguntas ("¿qué eventos tengo?")
            - Comandos/acciones ("crea un evento")
            - Información sobre terceros ("mi mamá se llama Rosa")
            - Datos temporales de eventos
            
            RESPONDE SOLO EN:
            True si HAY información personal nueva que DEBE extraerse.
            False si NO hay información personal nueva que extraer.
        """

        messages_to_send = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content)
        ]
        try:
            structured_llm = self.classifier_llm.with_structured_output(ExtractionDecision)
            result = structured_llm.invoke(messages_to_send)
            return result.should_extract
        
        except Exception as e:
            print(f"Error in LLM classifier: {e}")
            return False
    
    def should_extract_memory(self, messages: Sequence[BaseMessage]) -> bool:
        """
        Determina si hay información personal nueva usando 
        Enfoque híbrido: heurísticas + LLM
        """
        # get last human message
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None
        )
        # no human message
        if not last_human:
            return False
        
        # heuristic filter
        content_lower = last_human.content.lower()
        has_question_mark = "?" in last_human.content
        has_negative_keywords = any(negative_kw in content_lower for negative_kw in self.negative_keywords)
        has_positive_patterns = any(postitive_kw in content_lower for postitive_kw in self.positive_patterns)
        
        if has_question_mark or has_negative_keywords:
            return False
        
        if has_positive_patterns:
            return True
        
        return self._llm_should_extract(last_human.content)
    
    def extract_and_save(
        self, 
        state: AgentState, 
        config: RunnableConfig, 
        store: BaseStore
    ) -> dict:
        """
        Extrae información personal de la conversación y actualiza el store.
        SOLO se llama cuando hay nueva información potencial.
        """
        namespace = self.get_namespace(config)
        existing_items = store.search(namespace)
        
        # format existing memory for TrustCall [(key, trustcal_schema, value)]
        truscall_schema = 'UserProfile'
        existing_memories = [
            (item.key, truscall_schema, item.value) 
            for item in existing_items
        ] if existing_items else None
        
        # format messages for TrustCall
        clean_messages = [
            m for m in state['messages'] 
            if isinstance(m, (HumanMessage, AIMessage)) 
            and not (isinstance(m, AIMessage) and m.tool_calls)
        ]
        window_size = 4
        clean_messages = clean_messages[-window_size:] if len(clean_messages) >= window_size else clean_messages
        instruction = """Analiza la conversación y extrae SOLO información personal nueva o actualizada.
        Información válida: nombre, ubicación, hobbies, gustos, rutinas, preferencias.
        Si no hay información nueva, devuelve el perfil existente sin cambios."""
        messages = [SystemMessage(content=instruction)] + merge_message_runs(clean_messages)
    
        # call TrustCall extractor
        result = self.extractor.invoke({
            'messages': messages,
            'existing': existing_memories
        })
        
        # save results
        for r, rmeta in zip(result['responses'], result['response_metadata']):
            store.put(
                namespace,
                rmeta.get('json_doc_id', str(uuid.uuid4())),
                r.model_dump(mode='json')
            )
        
        return {"messages": []}  # No message

#  global del manager
memory_manager = MemoryManager(llm)

# ==================== LAYER 3: NODES ====================

def agent_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Nodo principal: Razona, decide y ejecuta herramientas.
    NO maneja memoria directamente.
    """
    # load user profile from store config
    user_profile = memory_manager.load_profile(store, config)
    
    # time
    cdmx_tz = ZoneInfo("America/Mexico_City")
    now = datetime.now(cdmx_tz).strftime("%A, %d de %B de %Y, %H:%M %p")
    
    # System prompt
    system_prompt = f"""
        Eres un Asistente de Productividad Personal inteligente.

        CONTEXTO TEMPORAL:
            - Ahora es: {now} (CDMX)

        PERFIL DEL USUARIO:
            {user_profile if user_profile else "No hay información de perfil aún."}

        INSTRUCCIONES:
            1. Responde de forma concisa y útil
            2. Usa las herramientas de calendario cuando sea necesario
            3. Sé proactivo y natural en tus respuestas

        IMPORTANTE: No menciones explícitamente que "guardas" información. Hazlo de forma transparente.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # invoke LLM with tools
    llm_with_tools = llm.bind_tools(CALENDAR_TOOLS)
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def memory_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Nodo de memoria: Ejecuta DESPUÉS del agente si detecta información personal nueva.
    Se ejecuta de forma asíncrona al flujo principal.
    """
    return memory_manager.extract_and_save(state, config, store)

def tools_node_wrapper(state: AgentState):
    """Wrapper para el ToolNode de calendario"""
    # Filtrar solo tool calls de calendario
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {"messages": []}
    
    calendar_calls = [
        tc for tc in last_message.tool_calls 
        if tc["name"] in CALENDAR_TOOL_NAMES
    ]
    
    if not calendar_calls:
        return {"messages": []}
    
    # only relevant tool calls
    temp_message = AIMessage(
        content=last_message.content,
        tool_calls=calendar_calls
    )
    temp_state = {"messages": state["messages"][:-1] + [temp_message]}
    
    # call tool node
    tool_node = ToolNode(CALENDAR_TOOLS)
    return tool_node.invoke(temp_state)

# ==================== ROUTING ====================

def route_after_agent(state: AgentState) -> List[str]:
    """
    Decide los siguientes pasos después del nodo agente.
    Puede retornar múltiples destinos (ejecución paralela).
    """
    last_message = state["messages"][-1]
    next_steps = []
    
    # see for calendar tool calls
    if last_message.tool_calls:
        calendar_calls = [
            tc for tc in last_message.tool_calls 
            if tc["name"] in CALENDAR_TOOL_NAMES
        ]
        if calendar_calls:
            next_steps.append("tools")
    
    # memory triger
    if memory_manager.should_extract_memory(state["messages"]):
        next_steps.append("memory")
    
    return next_steps if next_steps else [END]

# ==================== GRAPH ====================

def build_graph():
    """Construye el grafo de estado del agente"""
    builder = StateGraph(AgentState, config_schema=configuration.Configuration)
    
    # nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node_wrapper)
    builder.add_node("memory", memory_node)
    
    # edges
    builder.add_edge(START, "agent")
    
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "memory": "memory",
            END: END
        }
    )
    
    builder.add_edge("tools", "agent")    
    builder.add_edge("memory", END)
    
    return builder.compile()

# ==================== DEPLOYMENT ====================
g_assistant = build_graph()
