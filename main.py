import ollama
from pathlib import Path
import sys


class ProjectContextChat:
    def __init__(
        self, context_dir="context", max_total_tokens=8000, exclude_patterns=None
    ):
        self.context_dir = Path(context_dir).resolve()
        self.history_file = self.context_dir / "conversation_history.txt"
        self.max_total_tokens = max_total_tokens
        self.exclude_patterns = exclude_patterns or [
            "venv",
            "*.log",
            "__pycache__",
            "migrations",
        ]
        self.file_cache = {}

        # Validações iniciais
        if not self.context_dir.is_dir():
            raise ValueError(
                f"Diretório de contexto '{self.context_dir}' não encontrado"
            )

        # Detecta o projeto automaticamente
        self.project_root = self._detect_project()
        self.history_file.touch(exist_ok=True)

    def _detect_project(self):
        """Detecta automaticamente o diretório do projeto"""
        candidates = []
        for entry in self.context_dir.iterdir():
            if entry.is_dir() and entry.name != "conversation_history.txt":
                candidates.append(entry)

        if len(candidates) != 1:
            raise ValueError(
                f"Encontrados {len(candidates)} projetos em '{self.context_dir}'. "
                "Deve haver exatamente um diretório de projeto."
            )

        return candidates[0]

    def _walk_project_files(self):
        """Varre arquivos do projeto excluindo padrões indesejados"""
        files = []
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            if any(
                path.match(pattern)
                or any(excl in path.parts for excl in self.exclude_patterns)
                for pattern in self.exclude_patterns
            ):
                continue
            files.append(path)
        return files

    def _estimate_tokens(self, text):
        """Estimativa simples de tokens baseada em palavras"""
        return len(text.split())

    def _summarize_file(self, content, max_tokens):
        """Resume conteúdo de arquivos grandes"""
        tokens = content.split()
        if len(tokens) <= max_tokens:
            return content
        half = max_tokens // 2
        return (
            " ".join(tokens[:half])
            + "\n[... conteúdo resumido por limite de tokens ...]\n"
            + " ".join(tokens[-half:])
        )

    def _build_project_context(self):
        """Constrói contexto completo do projeto"""
        context = []
        total_tokens = 0
        files = self._walk_project_files()

        # Ordena por tipo de arquivo (prioridade para Python)
        priority_order = [".py", ".md", ".env", ".yml", ".toml"]
        files.sort(
            key=lambda x: (
                next(
                    (i for i, ext in enumerate(priority_order) if x.suffix == ext),
                    len(priority_order),
                ),
                str(x),
            )
        )

        for file_path in files:
            try:
                # Verificação de cache
                last_modified = file_path.stat().st_mtime
                if (
                    file_path in self.file_cache
                    and self.file_cache[file_path]["timestamp"] == last_modified
                ):
                    content = self.file_cache[file_path]["content"]
                    token_count = self.file_cache[file_path]["tokens"]
                else:
                    # Processa novo/alterado
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    token_count = self._estimate_tokens(content)
                    content = self._summarize_file(content, 500)
                    self.file_cache[file_path] = {
                        "timestamp": last_modified,
                        "content": content,
                        "tokens": token_count,
                    }

                # Adiciona ao contexto se couber
                if total_tokens + token_count > self.max_total_tokens:
                    break
                relative_path = file_path.relative_to(self.project_root)
                context.append(f"# {relative_path}\n{content}")
                total_tokens += token_count

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

        # Cria sumário do projeto
        project_summary = (
            f"Estrutura do projeto '{self.project_root.name}':\n"
            + "\n".join(f"- {f.relative_to(self.project_root)}" for f in files[:20])
            + "\n[...]\n"
            if len(files) > 20
            else ""
        )

        return f"{project_summary}\n\n" + "\n\n".join(context)

    def _load_history(self, max_tokens=2000):
        """Carrega histórico relevante"""
        try:
            lines = []
            token_count = 0
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    line_tokens = self._estimate_tokens(line)
                    if token_count + line_tokens > max_tokens:
                        break
                    lines.insert(0, line.strip())
                    token_count += line_tokens
            return "\n".join(lines)
        except Exception as e:
            print(f"Erro ao carregar histórico: {e}")
            return ""

    def chat(self, prompt):
        """Processa interação completa com streaming"""
        try:
            if prompt.lower() in ["sair", "exit"]:
                print("\nEncerrando a conversa.")
                return

            # Constrói contexto completo
            project_context = self._build_project_context()
            history = self._load_history()

            # Cria prompt estruturado
            full_prompt = (
                f"CONTEXTUALIZAÇÃO DO PROJETO:\n{project_context}\n\n"
                f"HISTÓRICO DA CONVERSA:\n{history}\n\n"
                f"NOVA PERGUNTA:\nUser: {prompt}\n\n"
                "Resposta detalhada (cite arquivos específicos):\nAssistant:"
            )

            # Envia para o modelo com streaming
            response = ""
            print("\nAssistente: ", end="", flush=True)
            for chunk in ollama.chat(
                model="qwen2.5-coder",
                messages=[{"role": "user", "content": full_prompt}],
                stream=True,
            ):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    response += token

            # Salva no histórico
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"User: {prompt}\n")
                f.write(f"Assistant: {response}\n")

        except Exception as e:
            print(f"\nErro durante a interação: {e}")


if __name__ == "__main__":
    try:
        chat = ProjectContextChat(
            context_dir="context",
            exclude_patterns=["venv", "*.log", "__pycache__", "migrations"],
        )
    except ValueError as e:
        print(f"Erro na configuração: {e}")
        sys.exit(1)

    print(f"\nAssistente iniciado para projeto: {chat.project_root.name}")
    print(f"Diretório de contexto: {chat.context_dir}")
    print(f"Arquivo de histórico: {chat.history_file}")
    print("Digite 'sair' ou 'exit' para encerrar.")

    while True:
        prompt = input("\n\nVocê: ")
        if prompt.lower() in ["sair", "exit"]:
            break
        chat.chat(prompt)
