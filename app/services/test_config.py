from app.services.llm_service import SmolLM


def test_system_prompt() -> None:
    llm = SmolLM()

    # Test 1: Default behavior
    print("\n--- Test 1: Default System Prompt ---")
    response1 = llm.generate("Who are you?")
    print(f"Response 1: {response1}")

    # Test 2: Custom personality
    print("\n--- Test 2: Pirate Personality ---")
    llm.update_config(
        system_prompt="You are a helpful assistant who speaks like a pirate."
    )
    response2 = llm.generate("Who are you?")
    print(f"Response 2: {response2}")

    # Test 3: Restrictions
    print("\n--- Test 3: Restrictions (No talk about space) ---")
    llm.update_config(
        system_prompt="You are a helpful assistant. You are absolutely not allowed to talk about space or planets. If asked, say you don't know."
    )
    response3 = llm.generate("What is Mars?")
    print(f"Response 3: {response3}")


if __name__ == "__main__":
    test_system_prompt()
