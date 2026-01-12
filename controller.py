from gpt2_small.predictor import GPTPredictor

class Controller:
    ENDOFTEXT = "<|endoftext|>"

    def __init__(self) -> None:
        self.predictor = GPTPredictor(weights_path="gpt2.weights.h5")
    
    def generate_text(self, prompt: str, max_tokens: int = 10) -> str:
        print('Generating text...')
        generated_text = self.predictor.predict(prompt, max_tokens=max_tokens)
        if self.ENDOFTEXT in generated_text:
            generated_text = generated_text.split(self.ENDOFTEXT)[0]
        else:
            print("Continuing generation to reach end-of-text token.")
            generated_text = self.generate_text(prompt, max_tokens=2*max_tokens)
        return generated_text

    def predict(self, materia: str, nivel: int) -> str:
        prompt = self._build_prompt(materia, nivel)
        generated_text = self.generate_text(prompt, max_tokens=50)
        return self._extract_problema(generated_text)
    
    def _build_prompt(self, materia: str, nivel: int) -> str:
        return f"Materia: {materia.upper()}\nNivel: {nivel}\nProblema:"
    
    def _extract_problema(self, generated_text: str) -> str:
        start_index = generated_text.find("Problema:") + len("Problema:")
        return generated_text[start_index:].strip()