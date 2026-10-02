"""Chat with the RescuePaws agent in the terminal. Run: python -m src.chat"""
import logging

from src.agent import ShelterAgent
from src.tools import ShelterTools


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")
    print("Loading RescuePaws assistant...")
    agent = ShelterAgent(ShelterTools())
    print("Ask me anything about adopting. Type 'quit' to exit.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue
        print(f"\nAssistant: {agent.chat(user_message)}\n")


if __name__ == "__main__":
    main()
