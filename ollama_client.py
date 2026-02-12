"""
Класс для работы с локальным API Ollama.
Поддерживает генерацию текста, чат, управление моделями и стриминг.
"""

import json
import requests
from typing import List, Dict, Optional, Iterator, Union


class OllamaClient:
    """Клиент для работы с API Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Инициализация клиента Ollama.
        
        Args:
            base_url: URL сервера Ollama (по умолчанию http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        model: str = "mistral",
        stream: bool = False,
        options: Optional[Dict] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        raw: bool = False,
        format: Optional[str] = None
    ) -> Union[Dict, Iterator[Dict]]:
        """
        Генерация текста на основе промпта.
        
        Args:
            prompt: Текст промпта для генерации
            model: Название модели (по умолчанию "mistral")
            stream: Если True, возвращает итератор с частями ответа
            options: Дополнительные параметры модели (temperature, top_p и т.д.)
            system: Системный промпт
            template: Шаблон для промпта
            context: Контекст предыдущих сообщений
            raw: Если True, не применяет шаблон к промпту
            format: Формат ответа (json и т.д.)
        
        Returns:
            Если stream=False: словарь с полным ответом
            Если stream=True: итератор словарей с частями ответа
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if context:
            payload["context"] = context
        if raw:
            payload["raw"] = raw
        if format:
            payload["format"] = format
        
        try:
            response = self.session.post(url, json=payload, stream=stream, timeout=300)
            response.raise_for_status()
            
            if stream:
                return self._parse_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при генерации текста: {str(e)}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "mistral",
        stream: bool = False,
        options: Optional[Dict] = None,
        format: Optional[str] = None
    ) -> Union[Dict, Iterator[Dict]]:
        """
        Чат с моделью.
        
        Args:
            messages: Список сообщений в формате [{"role": "user", "content": "..."}, ...]
            model: Название модели (по умолчанию "mistral")
            stream: Если True, возвращает итератор с частями ответа
            options: Дополнительные параметры модели
            format: Формат ответа (json и т.д.)
        
        Returns:
            Если stream=False: словарь с полным ответом
            Если stream=True: итератор словарей с частями ответа
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        
        try:
            response = self.session.post(url, json=payload, stream=stream, timeout=300)
            response.raise_for_status()
            
            if stream:
                return self._parse_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при чате: {str(e)}")
    
    def list_models(self) -> Dict:
        """
        Получить список установленных моделей.
        
        Returns:
            Словарь со списком моделей
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при получении списка моделей: {str(e)}")
    
    def pull_model(self, model_name: str, stream: bool = False) -> Union[Dict, Iterator[Dict]]:
        """
        Загрузить модель из репозитория.
        
        Args:
            model_name: Название модели для загрузки (например, "mistral:7b")
            stream: Если True, возвращает итератор с прогрессом загрузки
        
        Returns:
            Если stream=False: словарь с результатом загрузки
            Если stream=True: итератор словарей с прогрессом загрузки
        """
        url = f"{self.base_url}/api/pull"
        
        payload = {
            "name": model_name,
            "stream": stream
        }
        
        try:
            response = self.session.post(url, json=payload, stream=stream, timeout=600)
            response.raise_for_status()
            
            if stream:
                return self._parse_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при загрузке модели: {str(e)}")
    
    def generate_json(
        self,
        prompt: str,
        model: str = "mistral",
        json_schema: Optional[Union[Dict, str]] = None,
        context: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        options: Optional[Dict] = None,
        validate: bool = True
    ) -> Dict:
        """
        Автоматическая генерация JSON с контекстом.
        
        Args:
            prompt: Промпт для генерации JSON
            model: Название модели (по умолчанию "mistral")
            json_schema: JSON схема в виде словаря или строки (опционально)
            context: Контекст предыдущих сообщений в формате [{"role": "...", "content": "..."}]
            system: Системный промпт с инструкциями по генерации JSON
            options: Дополнительные параметры модели
            validate: Если True, проверяет валидность JSON ответа
        
        Returns:
            Словарь с распарсенным JSON ответом
        
        Example:
            >>> client = OllamaClient()
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "number"}
            ...     }
            ... }
            >>> result = client.generate_json(
            ...     "Создай JSON с именем и возрастом человека",
            ...     json_schema=schema
            ... )
        """
        # Формируем системный промпт для генерации JSON
        if system is None:
            system_prompt = "Ты - помощник, который генерирует только валидный JSON. Отвечай только JSON, без дополнительных комментариев или текста."
        else:
            system_prompt = system
        
        # Добавляем JSON схему в промпт, если она указана
        if json_schema:
            if isinstance(json_schema, dict):
                schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2)
            else:
                schema_str = str(json_schema)
            
            enhanced_prompt = f"""{prompt}

Сгенерируй JSON согласно следующей схеме:
{schema_str}

Важно: верни только валидный JSON без дополнительного текста."""
        else:
            enhanced_prompt = prompt
        
        # Если есть контекст, используем chat API, иначе generate
        if context:
            messages = context.copy()
            messages.append({"role": "user", "content": enhanced_prompt})
            
            # Добавляем системное сообщение, если его нет
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "format": "json",
                "stream": False
            }
            
            if options:
                payload["options"] = options
            
            url = f"{self.base_url}/api/chat"
        else:
            payload = {
                "model": model,
                "prompt": enhanced_prompt,
                "system": system_prompt,
                "format": "json",
                "stream": False
            }
            
            if options:
                payload["options"] = options
            
            url = f"{self.base_url}/api/generate"
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # Извлекаем JSON из ответа
            if context:
                # Для chat API ответ в поле message.content
                json_text = result.get("message", {}).get("content", "")
            else:
                # Для generate API ответ в поле response
                json_text = result.get("response", "")
            
            # Парсим JSON
            if validate and json_text:
                try:
                    parsed_json = json.loads(json_text.strip())
                    return parsed_json
                except json.JSONDecodeError as e:
                    raise OllamaAPIError(f"Ответ не является валидным JSON: {str(e)}\nПолученный текст: {json_text[:200]}")
            else:
                return result
                
        except json.JSONDecodeError as e:
            raise OllamaAPIError(f"Ошибка парсинга JSON ответа: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при генерации JSON: {str(e)}")
    
    def show_model_info(self, model_name: str) -> Dict:
        """
        Получить информацию о модели.
        
        Args:
            model_name: Название модели
        
        Returns:
            Словарь с информацией о модели
        """
        url = f"{self.base_url}/api/show"
        
        payload = {
            "name": model_name
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Ошибка при получении информации о модели: {str(e)}")
    
    def _parse_stream_response(self, response: requests.Response) -> Iterator[Dict]:
        """
        Парсинг стримингового ответа (SSE формат).
        
        Args:
            response: Response объект с включенным стримингом
        
        Yields:
            Словари с частями ответа
        """
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.strip():
                            yield json.loads(decoded_line)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise OllamaAPIError(f"Ошибка при парсинге стримингового ответа: {str(e)}")


class OllamaAPIError(Exception):
    """Исключение для ошибок API Ollama."""
    pass

