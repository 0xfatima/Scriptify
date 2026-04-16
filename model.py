from model_singleton import generate_writing_assist


def run_model(text, mode="General") -> str:
    return generate_writing_assist(text=text, mode=mode)