"""RescuePaws agent: the LLM decides which tool to call, we run it, and loop."""
import json
import logging

from src.llm import chat_with_tools
from src.tools import TOOL_SCHEMAS, ShelterTools

logger = logging.getLogger("rescuepaws.agent")

SYSTEM_PROMPT = """You are the RescuePaws assistant for a volunteer-run animal shelter. You help adopters find animals and understand shelter policies.

Rules:
- Use tools for every fact about animals or shelter policy. Never answer those from your own knowledge.
- Only state animal facts that appear in tool results. If a field is "unknown" or missing, say it is not recorded. Never guess health, temperament, or compatibility with children, dogs, or cats.
- If an animal's notes mention a concern that conflicts with another field, mention both and suggest the adopter ask shelter staff.
- When someone describes what they want, call match_animals. Fill has_children, has_dogs, and has_cats from what they say about their household. Present matches with their reasons and unknowns.
- If someone asks for a recommendation but hasn't said what kind of animal they want or who lives in their home (children, dogs, cats), ask about that first. Compatibility can't be checked without it.
- If a name matches more than one animal, list them with their IDs and ask which one the user means.
- When the user refers to an animal from earlier in the conversation ("the first one", "that dog"), use its animal ID from the earlier results with get_animal_profile instead of searching by name.
- For policy questions, call answer_policy_question and relay its answer and sources. If it did not find the answer, say the policy documents don't cover it.
- Do not give medical advice. For health concerns, suggest contacting a veterinarian.
- Always include animal IDs (like RP-0009) when you mention specific animals. Keep answers concise."""

MAX_STEPS = 6


class ShelterAgent:
    def __init__(self, tools: ShelterTools, llm=chat_with_tools):
        self.tools = tools
        self.llm = llm
        self.messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_log: list[dict] = []  # every tool call, for debugging and evaluation

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_STEPS):
            message = self.llm(self.messages, TOOL_SCHEMAS)
            self.messages.append(message)

            if not message.tool_calls:
                return message.content  # no more tools needed: this is the final answer

            for call in message.tool_calls:
                name = call.function.name
                arguments = json.loads(call.function.arguments)
                result = self.tools.execute(name, arguments)

                logger.info("tool=%s args=%s result=%s", name, arguments, json.dumps(result)[:300])
                self.tool_log.append({"tool": name, "arguments": arguments, "result": result})
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })

        return "Sorry, I couldn't complete that request. Please try rephrasing it."
