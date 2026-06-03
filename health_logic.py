from dataclasses import dataclass, field
import json
import os
from typing import Any


HEALTH_GUIDANCE_DOCUMENTS = [
    "General self-care: give educational guidance only. Do not diagnose, prescribe, name medicines, suggest dosages, or replace a healthcare professional. Safe advice includes rest, hydration, sleep, nutrition, steam inhalation if comfortable, warm salt-water gargles, avoiding strenuous activity, and monitoring symptoms.",
    "Respiratory symptoms: fever, cough, sore throat, runny nose, fatigue, and body aches may be consistent with a common viral illness, throat irritation, seasonal infection, or respiratory irritation.",
    "Digestive symptoms: nausea, vomiting, and diarrhea may be consistent with short-term stomach upset, food-related irritation, or digestive infection. Watch for dehydration, faintness, confusion, or worsening weakness.",
    "Warning signs: breathing difficulty, chest pain, severe symptoms, fainting, confusion, dehydration, high or persistent fever, symptoms worsening after several days, or symptoms lasting more than a week should prompt medical care. Breathing difficulty or chest pain is urgent.",
    "Existing conditions: diabetes, hypertension, asthma, heart disease, pregnancy, kidney disease, or immune system conditions can increase risk. People with these conditions should seek professional advice earlier if symptoms continue, worsen, or become moderate or severe.",
]

WARNING_SYMPTOMS = {"chest pain", "dizziness"}
HIGHER_RISK_CONDITIONS = {
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "pregnancy",
    "kidney disease",
    "immune system condition",
}


@dataclass
class HealthInput:
    symptoms: list[str]
    conditions: list[str]
    age: int | None
    duration: str
    severity: str
    breathing_difficulty: bool
    details: str = ""

    def to_prompt_context(self) -> str:
        return (
            f"Symptoms: {', '.join(self.symptoms)}\n"
            f"Existing conditions: {', '.join(self.conditions) or 'None'}\n"
            f"Age: {self.age if self.age is not None else 'Not provided'}\n"
            f"Duration: {self.duration}\n"
            f"Severity: {self.severity}\n"
            f"Breathing difficulty: {'Yes' if self.breathing_difficulty else 'No'}\n"
            f"Other details: {self.details or 'None'}"
        )


@dataclass
class GeminiConfig:
    credentials_path: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    )
    project: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    model: str = "gemini-2.5-flash"


def analyze_health_info(info: HealthInput) -> dict:
    symptom_set = {item.lower() for item in info.symptoms}
    condition_set = {item.lower() for item in info.conditions}
    urgent = info.breathing_difficulty or "chest pain" in symptom_set
    return {
        "urgent": urgent,
        "serious": urgent
        or info.severity == "Severe"
        or bool(symptom_set & WARNING_SYMPTOMS)
        or (info.age is not None and (info.age < 5 or info.age >= 65)),
        "cautious": bool(condition_set & HIGHER_RISK_CONDITIONS)
        or info.duration in {"More than 1 week", "More than 2 weeks"}
        or info.severity == "Moderate",
    }


class DynamicHealthChatbot:
    def __init__(self, max_history_messages: int = 10) -> None:
        self.max_history_messages = max_history_messages
        self.session_id = "healthguide-session"
        self.config = GeminiConfig()
        self._store: dict[str, Any] = {}
        self._chain = None
        self._chain_error = ""
        self._config_key = ""

    def configure(self, config: GeminiConfig) -> None:
        config_key = "|".join(
            [config.credentials_path, config.project, config.location, config.model]
        )
        if config_key == self._config_key:
            return

        self.config = config
        self._config_key = config_key
        self._chain = None
        self._chain_error = ""

        try:
            self._chain = self._build_chain()
        except Exception as exc:
            self._chain_error = str(exc)

    def ask(self, user_message: str, health_info: HealthInput) -> str:
        user_message = user_message.strip()
        if not user_message:
            return "Please type a health question or describe what changed."

        if self._chain is None:
            return self._missing_chain_message()

        safety_flags = analyze_health_info(health_info)

        result = self._chain.invoke(
            {
                "input": (
                    f"Current health information:\n{health_info.to_prompt_context()}\n\n"
                    f"Safety flags detected by app: {safety_flags}\n\n"
                    f"User message:\n{user_message}"
                )
            },
            config={"configurable": {"session_id": self.session_id}},
        )
        answer = result.get("answer", str(result)) if isinstance(result, dict) else str(result)
        return self._add_safety_note(answer)

    def clear_history(self) -> None:
        history = self._store.get(self.session_id)
        if history:
            history.clear()

    def _build_chain(self) -> Any:
        self._prepare_gcp_credentials()

        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.documents import Document
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_google_vertexai import ChatVertexAI
        from pydantic import Field

        max_messages = self.max_history_messages

        class TenMessageHistory(BaseChatMessageHistory):
            messages: list[BaseMessage] = Field(default_factory=list)

            def add_message(self, message: BaseMessage) -> None:
                self.messages.append(message)
                self.messages = self.messages[-max_messages:]

            def clear(self) -> None:
                self.messages = []

        class KeywordHealthRetriever(BaseRetriever):
            documents: list[Document]

            def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                words = {
                    word.strip(".,:;!?()[]{}").lower()
                    for word in query.split()
                    if len(word.strip(".,:;!?()[]{}")) > 2
                }
                ranked = []
                for document in self.documents:
                    text = document.page_content.lower()
                    ranked.append((sum(1 for word in words if word in text), document))
                ranked.sort(key=lambda item: item[0], reverse=True)
                matches = [document for score, document in ranked if score > 0][:4]
                return matches or self.documents[:4]

        documents = [Document(page_content=text) for text in HEALTH_GUIDANCE_DOCUMENTS]
        retriever = KeywordHealthRetriever(documents=documents)
        llm = ChatVertexAI(
            model=self.config.model,
            project=self.config.project or None,
            location=self.config.location,
            temperature=0.2,
        )

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given chat history and the latest health question, create a "
                    "standalone retrieval query. Do not answer the user.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are HealthGuide, a careful health self-care chatbot. Use the "
                    "retrieved context to answer. Do not diagnose. Never recommend "
                    "medicines, drug names, dosages, or prescriptions. Only provide "
                    "non-medication advice: rest, hydration, sleep, nutrition, steam "
                    "inhalation, warm salt-water gargles, avoiding strenuous activity, "
                    "and monitoring symptoms. Escalate warning signs clearly.\n\n"
                    "Retrieved context:\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self._store:
                self._store[session_id] = TenMessageHistory()
            return self._store[session_id]

        return RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _prepare_gcp_credentials(self) -> None:
        credentials_path = self.config.credentials_path.strip()
        if credentials_path:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"GCP JSON key not found: {credentials_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        if not self.config.project and credentials_path:
            with open(credentials_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            self.config.project = data.get("project_id", "")

    def _missing_chain_message(self) -> str:
        return (
            "Gemini is not configured yet, so I cannot generate a dynamic answer.\n\n"
            "Add your GCP service-account JSON key path in the GCP JSON key path field, "
            "confirm the project/location/model values, and install the packages in "
            "requirements.txt.\n\n"
            "Do not paste the JSON contents into the app. Paste the file path only, "
            "for example: C:\\Users\\abhis\\keys\\healthguide-gemini.json"
            + (f"\n\nSetup detail: {self._chain_error}" if self._chain_error else "")
        )

    def _add_safety_note(self, answer: str) -> str:
        if "No medicine advice:" in answer:
            return answer
        return (
            answer.rstrip()
            + "\n\nNo medicine advice: This chatbot does not recommend drug names, dosages, or prescriptions."
        )
