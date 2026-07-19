import logging
from .rag_chain import RAGChain, create_rag_chain
#from config import config
from typing import Optional

logger = logging.getLogger(__name__)


class ChatSession:
    """Simple chat session"""

    def __init__(self, rag_chain: RAGChain):
        self.rag = rag_chain
        self.history = []

    def ask(self, question: str) -> dict:
        """Ask a question and track history"""
        response = self.rag.ask(question, return_sources=True)

        self.history.append(
            {
                "question": question,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
            }
        )

        return response

    def show_history(self, limit: int = 5):
        """Show recent conversation history"""
        if not self.history:
            print("\n📝 No conversation history yet.\n")
            return

        print(f"\n📝 Last {min(limit, len(self.history))} messages:\n")
        for i, turn in enumerate(self.history[-limit:], 1):
            print(f"{i}. Q: {turn['question'][:100]}")
            print(f"   A: {turn['answer'][:150]}...\n")

    def clear(self):
        """Clear history"""
        self.history = []
        print("✅ History cleared.\n")


def run_chat(
    config,
    retriever_strategy: str = "hybrid",
    reranker_type: Optional[str] = None,
    k: int = 3,
    **kwargs,
):
    """
    Run interactive chat

    Args:
        config: PipelineConfig
        retriever_strategy: Retrieval strategy
        reranker_type: Reranker type
        k: Number of documents
    """
    print("\n" + "=" * 60)
    print("🤖 RAG Chat Assistant")
    print("=" * 60)
    print(f"\nUsing: {getattr(config, 'llm', 'llama3:8b')}")
    print(f"Strategy: {retriever_strategy}")
    print(f"Documents per query: {k}\n")
    print("Commands:")
    print("  /help    - Show this message")
    print("  /history - Show conversation history")
    print("  /clear   - Clear history")
    print("  /quit    - Exit")
    print("\n" + "=" * 60 + "\n")

    try:
        # Create RAG chain
        rag = create_rag_chain(
            config=config,
            retriever_strategy=retriever_strategy,
            reranker_type=reranker_type,
            k=k,
            **kwargs,
        )

        # Create session
        session = ChatSession(rag)

        while True:
            try:
                # Get input
                user_input = input("❓ You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == "/quit":
                    print("\n👋 Goodbye!\n")
                    break
                elif user_input.lower() == "/help":
                    print("\nCommands:")
                    print("  /help    - Show this message")
                    print("  /history - Show conversation history")
                    print("  /clear   - Clear history")
                    print("  /quit    - Exit\n")
                    continue
                elif user_input.lower() == "/history":
                    session.show_history()
                    continue
                elif user_input.lower() == "/clear":
                    session.clear()
                    continue

                # Process question
                print("\n🤔 Thinking...")
                response = session.ask(user_input)

                if response.get("success"):
                    print(f"\n🤖 Assistant: {response['answer']}\n")

                    # Show sources
                    sources = response.get("sources", [])
                    if sources:
                        print("📚 Sources:")
                        for source in sources[:3]:
                            source_name = source.get("source", "Unknown")
                            print(f"   • {source_name}")
                        print()
                else:
                    print(f"\n❌ {response.get('answer', 'Unknown error')}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                logger.error(f"Chat error: {e}")

    except Exception as e:
        print(f"\n❌ Failed to start chat: {e}")
        logger.error(f"Failed to start chat: {e}")
