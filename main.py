import os

import ollama


class QwenChat:
    """
    Classe para interagir com o modelo Qwen2.5-Coder via Ollama, com streaming de respostas e histórico persistente.
    """

    def __init__(self, history_file="conversation_history.txt"):
        """
        Inicializa a instância do chat.

        Args:
            history_file (str): Nome do arquivo onde o histórico será salvo/carregado.
        """
        self.history_file = history_file

    def save_to_history(self, line):
        """
        Adiciona uma nova linha ao arquivo de histórico.

        Args:
            line (str): Linha a ser adicionada ao histórico.
        """
        try:
            with open(self.history_file, "a", encoding="utf-8") as file:
                file.write(line + "\n")
        except Exception as e:
            print(f"Erro ao salvar no histórico: {e}")

    def load_relevant_history(self, max_tokens=2000):
        """
        Carrega o histórico mais recente do arquivo, respeitando o limite de tokens.

        Args:
            max_tokens (int): Número máximo de tokens permitidos no contexto.

        Returns:
            str: Histórico concatenado como uma string.
        """
        try:
            history_lines = []
            token_count = 0

            # Estimativa simples de tokens (ajuste conforme necessário)
            def estimate_tokens(text):
                return len(text.split())

            with open(self.history_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in reversed(lines):
                    line_tokens = estimate_tokens(line)
                    if token_count + line_tokens > max_tokens:
                        break
                    history_lines.insert(0, line.strip())
                    token_count += line_tokens

            return "\n".join(history_lines)
        except Exception as e:
            print(f"Erro ao carregar o histórico: {e}")
            return ""

    def chat(self, prompt, reasoning=False):
        """
        Envia o prompt para o modelo via Ollama, exibindo a resposta em streaming.

        Args:
            prompt (str): Entrada do usuário.
            reasoning (bool): Se True, solicita que o modelo explique seu raciocínio.

        Returns:
            str: Resposta completa do assistente.
        """
        try:
            # Verifica se o comando é especial
            if prompt.lower() in ["sair", "exit"]:
                print("\nEncerrando a conversa.")
                return None

            # Salva a entrada do usuário no histórico
            self.save_to_history(f"User: {prompt}")

            # Carrega o histórico relevante
            relevant_history = self.load_relevant_history(max_tokens=2000)

            # Cria o prompt completo
            if reasoning:
                full_prompt = (
                    relevant_history + f"\nUser: {prompt}\n"
                    "Por favor, explique seu raciocínio passo a passo antes de fornecer a resposta final.\n"
                    "Assistant: "
                )
            else:
                full_prompt = relevant_history + f"\nUser: {prompt}\nAssistant: "

            # Inicia o streaming da resposta
            response_generator = ollama.chat(
                model="qwen2.5-coder",  # Modelo atualizado para qwen2.5-coder
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,  # Ativa o streaming
            )

            # Processa e exibe a resposta em tempo real
            assistant_response = ""
            print("\nAssistente: ", end="", flush=True)
            for chunk in response_generator:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)  # Exibe o token imediatamente
                    assistant_response += token

            # Salva a resposta completa no histórico
            self.save_to_history(f"Assistant: {assistant_response}")

            print()  # Quebra de linha após a resposta
            return assistant_response

        except Exception as e:
            print(f"\nErro durante a interação: {e}")
            return None


# Exemplo de uso
if __name__ == "__main__":
    print(
        "Bem-vindo ao Qwen2.5-Coder! Digite 'sair' ou 'exit' para encerrar a conversa."
    )
    chatbot = QwenChat()

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]:
            break

        # Ativa o modo de raciocínio se o usuário pedir
        if "explique" in user_input.lower() or "raciocínio" in user_input.lower():
            chatbot.chat(user_input, reasoning=True)
        else:
            chatbot.chat(user_input)
