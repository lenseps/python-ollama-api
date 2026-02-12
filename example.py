from ollama_client import OllamaClient, OllamaAPIError


def example_basic_generate():
    """Пример базовой генерации текста."""
    print("=== Пример базовой генерации ===")
    
    client = OllamaClient()
    
    try:
        response = client.generate(
            prompt="Расскажи мне короткую историю о программировании",
            model="mistral"
        )
        print(f"Ответ: {response.get('response', '')}\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_streaming_generate():
    """Пример генерации со стримингом."""
    print("=== Пример генерации со стримингом ===")
    
    client = OllamaClient()
    
    try:
        stream = client.generate(
            prompt="Опиши искусственный интеллект в трех предложениях",
            model="mistral",
            stream=True
        )
        
        print("Ответ (стриминг): ", end="", flush=True)
        for chunk in stream:
            if "response" in chunk:
                print(chunk["response"], end="", flush=True)
        print("\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_chat():
    """Пример чата с моделью."""
    print("=== Пример чата ===")
    
    client = OllamaClient()
    
    messages = [
        {"role": "user", "content": "Привет! Как дела?"}
    ]
    
    try:
        response = client.chat(
            messages=messages,
            model="llama2"
        )
        
        assistant_message = response.get("message", {}).get("content", "")
        print(f"Ассистент: {assistant_message}\n")
        
        messages.append(response["message"])
        messages.append({"role": "user", "content": "Расскажи мне шутку"})
        
        response = client.chat(
            messages=messages,
            model="llama2"
        )
        print(f"Ассистент: {response.get('message', {}).get('content', '')}\n")
        
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_list_models():
    """Пример получения списка моделей."""
    print("=== Список моделей ===")
    
    client = OllamaClient()
    
    try:
        models_response = client.list_models()
        models = models_response.get("models", [])
        
        print(f"Установлено моделей: {len(models)}")
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0)
            print(f"  - {name} ({size / 1024 / 1024 / 1024:.2f} GB)")
        print()
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_model_info():
    """Пример получения информации о модели."""
    print("=== Информация о модели ===")
    
    client = OllamaClient()
    
    try:
        info = client.show_model_info("mistral")
        print(f"Модель: {info.get('modelfile', '')[:100]}...")
        print(f"Параметры: {info.get('parameters', '')[:100]}...\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_json_generate():
    """Пример генерации JSON без схемы."""
    print("=== Пример генерации JSON (без схемы) ===")
    
    client = OllamaClient()
    
    try:
        result = client.generate_json(
            prompt="Создай JSON объект с информацией о продукте: название, цена, категория",
            model="mistral"
        )
        print(f"Результат: {result}\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_json_with_schema():
    """Пример генерации JSON со схемой."""
    print("=== Пример генерации JSON со схемой ===")
    
    client = OllamaClient()
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Имя человека"},
            "age": {"type": "number", "description": "Возраст"},
            "email": {"type": "string", "description": "Email адрес"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Список навыков"
            }
        },
        "required": ["name", "age", "email"]
    }
    
    try:
        result = client.generate_json(
            prompt="Создай JSON профиль программиста на Python",
            model="llama2",
            json_schema=schema
        )
        print(f"Результат: {result}\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


def example_json_with_context():
    """Пример генерации JSON с контекстом."""
    print("=== Пример генерации JSON с контекстом ===")
    
    client = OllamaClient()
    
    context = [
        {"role": "user", "content": "Меня зовут Иван, мне 30 лет, я работаю программистом"},
        {"role": "assistant", "content": "Приятно познакомиться, Иван!"},
    ]
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "profession": {"type": "string"}
        }
    }
    
    try:
        result = client.generate_json(
            prompt="Создай JSON объект с информацией обо мне из нашего разговора",
            model="llama2",
            json_schema=schema,
            context=context
        )
        print(f"Результат: {result}\n")
    except OllamaAPIError as e:
        print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    print("Примеры использования OllamaClient\n")
    
    client = OllamaClient()
    try:
        client.list_models()
        print("API Ollama доступен!\n")
    except OllamaAPIError as e:
        print(f"Ошибка подключения к API Ollama: {e}")
        print("Убедитесь, что Ollama запущена на http://localhost:11434\n")
        exit(1)
    
    example_list_models()
    example_basic_generate()
    example_streaming_generate()
    example_chat()
    example_model_info()
    example_json_generate()
    example_json_with_schema()
    example_json_with_context()
    
    print("Для запуска других примеров раскомментируйте соответствующие функции.")

