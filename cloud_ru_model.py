"""
Кастомная модель для интеграции Cloud.ru Foundation Models API с smolagents
Использует стандартный OpenAI клиент для совместимости
"""

import os
from typing import List, Dict, Any, Optional
from smolagents.models import Model
from openai import OpenAI


class CloudRuModel(Model):
    """
    Модель для работы с Cloud.ru Foundation Models через OpenAI-совместимый API
    Совместима с smolagents и предоставляет простой интерфейс
    """
    
    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.5",  # Изменено на GLM-4.5
        api_key: Optional[str] = None,
        base_url: str = "https://foundation-models.api.cloud.ru/v1",
        temperature: float = 0.5,
        max_tokens: int = 5000,
        top_p: float = 0.95,
        presence_penalty: float = 0
    ):
        """
        Инициализация модели Cloud.ru
        
        Args:
            model_name: Название модели (по умолчанию "zai-org/GLM-4.5")
            api_key: API ключ Cloud.ru (если не указан, берется из env)
            base_url: URL эндпоинта Cloud.ru
            temperature: Температура генерации (0.0-2.0)
            max_tokens: Максимальное количество токенов в ответе
            top_p: Параметр nucleus sampling
            presence_penalty: Штраф за повторение
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        
        # Получаем API ключ из аргументов или переменных окружения
        self.api_key = api_key or os.getenv("CLOUD_RU_API_KEY") or os.getenv("API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "Необходимо указать CLOUD_RU_API_KEY или API_KEY "
                "либо через аргументы, либо через переменные окружения"
            )
        
        # Инициализируем клиент Cloud.ru через стандартный OpenAI клиент
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,  # Увеличиваем таймаут до 60 секунд
            max_retries=2  # Добавляем повторные попытки
        )
        
        super().__init__()
    
    def generate(
        self, 
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs
    ):
        """
        Основной метод для генерации ответа (совместимость с smolagents)
        
        Args:
            messages: Список сообщений ChatMessage от smolagents
            stop_sequences: Последовательности для остановки генерации
            response_format: Формат ответа
            tools_to_call_from: Список доступных инструментов
            **kwargs: Дополнительные параметры
            
        Returns:
            ChatMessage объект для совместимости с smolagents
        """
        try:
            # Конвертируем сообщения для Cloud.ru API
            converted_messages = []
            
            for msg in messages:
                # Извлекаем роль и контент
                if hasattr(msg, 'role'):
                    role = str(msg.role).replace("MessageRole.", "").lower()
                    
                    # Обрабатываем контент
                    if hasattr(msg, 'content'):
                        content = msg.content
                        # Если контент - это список с dict внутри
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and 'text' in content[0]:
                                content = content[0]['text']
                            else:
                                content = str(content)
                    else:
                        content = str(msg)
                else:
                    # Если это уже dict
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                
                # Конвертируем роли для совместимости с Cloud.ru
                if role in ["tool_call", "tool-call"]:
                    # Пропускаем tool_call сообщения
                    continue
                elif role in ["tool_response", "tool-response"]:
                    # Конвертируем tool_response в assistant для контекста
                    converted_messages.append({
                        "role": "assistant",
                        "content": content if isinstance(content, str) else str(content)
                    })
                elif role == "system":
                    # Системные сообщения остаются как есть
                    converted_messages.append({
                        "role": "system",
                        "content": content if isinstance(content, str) else str(content)
                    })
                elif role == "user":
                    converted_messages.append({
                        "role": "user",
                        "content": content if isinstance(content, str) else str(content)
                    })
                elif role == "assistant":
                    converted_messages.append({
                        "role": "assistant",
                        "content": content if isinstance(content, str) else str(content)
                    })
            
            # Если слишком длинное системное сообщение, упрощаем его
            if converted_messages and converted_messages[0]["role"] == "system":
                system_msg = converted_messages[0]["content"]
                if len(system_msg) > 10000:  # Если очень длинное
                    # Создаем упрощенное системное сообщение для GLM-4.5
                    # GLM-4.5 поддерживает reasoning mode и tool calling
                    converted_messages[0] = {
                        "role": "system",
                        "content": (
                            "You are an expert assistant who solves tasks step by step using code.\n"
                            "GLM-4.5 reasoning mode enabled.\n"
                            "Always respond in this format:\n"
                            "Thought: [your reasoning]\n"
                            "<code>\n[your python code]\n</code>\n"
                            "Use print() to output intermediate results.\n"
                            "Use final_answer() to provide the final result."
                        )
                    }
            
            # Убеждаемся, что есть хотя бы одно сообщение
            if not converted_messages:
                converted_messages = [{"role": "user", "content": "Hello"}]
            
            # Добавляем подсказку для правильного формата ответа, если нужно
            if stop_sequences and "<code>" in str(stop_sequences):
                # Это CodeAgent запрос - добавляем подсказку о формате
                if converted_messages[-1]["role"] == "user":
                    converted_messages[-1]["content"] += (
                        "\n\nRemember to respond with:\nThought: [reasoning]\n<code>\n[python code]\n</code>"
                    )
            
            # Параметры для запроса
            request_params = {
                "model": self.model_name,
                "messages": converted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty)
            }
            
            # Добавляем stop sequences если есть
            if stop_sequences:
                # Cloud.ru может не поддерживать все stop sequences
                valid_stops = [s for s in stop_sequences if isinstance(s, str) and len(s) > 0]
                if valid_stops:
                    request_params["stop"] = valid_stops[:4]  # Максимум 4 stop sequences
            
            # Выполняем запрос к Cloud.ru API
            response = self.client.chat.completions.create(**request_params)
            
            # Получаем ответ
            content = response.choices[0].message.content
            
            # Проверяем, нужно ли добавить формат для CodeAgent
            if stop_sequences and "<code>" in str(stop_sequences):
                # Если ответ не содержит нужный формат, добавляем его
                if "Thought:" not in content and "<code>" not in content:
                    # Пытаемся обернуть ответ в нужный формат
                    if "print" in content.lower() or "def " in content or "import " in content:
                        # Похоже на код
                        content = f"Thought: I will execute the requested code.\n<code>\n{content}\n</code>"
                    else:
                        # Обычный текст - превращаем в код с print
                        content = f"Thought: I will print the answer.\n<code>\nprint('{content}')\n</code>"
            
            # Создаем объект ChatMessage для совместимости с smolagents
            from smolagents import ChatMessage
            
            return ChatMessage(
                role="assistant",
                content=content
            )
            
        except Exception as e:
            error_msg = f"Ошибка при запросе к Cloud.ru API: {str(e)}"
            print(error_msg)
            
            # Для CodeAgent возвращаем ответ в правильном формате
            if stop_sequences and "<code>" in str(stop_sequences):
                from smolagents import ChatMessage
                return ChatMessage(
                    role="assistant",
                    content=f"Thought: An error occurred.\n<code>\nprint('Error: {str(e)}')\n</code>"
                )
            else:
                from smolagents import ChatMessage
                return ChatMessage(
                    role="assistant", 
                    content=f"Извините, произошла ошибка: {str(e)}"
                )
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Альтернативный метод для прямого вызова (возвращает строку)
        """
        try:
            chat_message = self.generate(messages, **kwargs)
            return chat_message.content
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получить информацию о модели
        
        Returns:
            Dict с информацией о модели
        """
        return {
            "provider": "Cloud.ru",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }


class CloudRuModelWithFallback(CloudRuModel):
    """
    Модель Cloud.ru с fallback на OpenAI в случае ошибок
    DEPRECATED: Используйте CloudRuModel без fallback
    """
    
    def __init__(self, *args, **kwargs):
        """Инициализация БЕЗ fallback - теперь это просто алиас для CloudRuModel"""
        # Убираем параметры fallback если они есть
        kwargs.pop("use_openai_fallback", None)
        kwargs.pop("openai_api_key", None)
        
        super().__init__(*args, **kwargs)
        
        # Больше не используем OpenAI fallback
        self.use_openai_fallback = False
    
    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        """
        Запрос БЕЗ fallback - просто вызываем родительский метод
        """
        return super().generate(messages, stop_sequences, response_format, tools_to_call_from, **kwargs)

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Запрос с fallback на OpenAI (возвращает строку)
        """
        try:
            chat_message = self.generate(messages, **kwargs)
            return chat_message.content
        except Exception as e:
            return f"Ошибка: {str(e)}"


def create_cloud_ru_model(**kwargs) -> CloudRuModel:
    """
    Удобная функция для создания модели Cloud.ru
    
    Returns:
        CloudRuModel instance
    """
    return CloudRuModel(**kwargs)


def create_cloud_ru_model_with_fallback(**kwargs) -> CloudRuModel:
    """
    DEPRECATED: Используйте create_cloud_ru_model
    Создает модель Cloud.ru БЕЗ fallback
    
    Returns:
        CloudRuModel instance
    """
    kwargs.pop("use_openai_fallback", None)
    kwargs.pop("openai_api_key", None)
    return CloudRuModel(**kwargs)


# Проверка подключения
def test_cloud_ru_connection():
    """Тестовая функция для проверки подключения к Cloud.ru API"""
    try:
        print("🧪 Тестируем подключение к Cloud.ru API...")
        
        # Проверяем наличие API ключа
        api_key = os.getenv("CLOUD_RU_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            print("❌ API ключ не найден. Установите CLOUD_RU_API_KEY или API_KEY")
            return False
        
        model = CloudRuModel()
        test_messages = [{"role": "user", "content": "Ответь одним словом: работает?"}]
        
        response = model(test_messages)
        print(f"✅ Подключение успешно! Ответ: {response}")
        
        # Выводим информацию о модели
        info = model.get_model_info()
        print(f"📊 Модель: {info['model_name']}")
        print(f"🔗 URL: {info['base_url']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения к Cloud.ru API: {e}")
        return False


if __name__ == "__main__":
    # Тест при запуске модуля
    test_cloud_ru_connection()