# Workshop: Multi-Agent система поиска лучшего курса валют

## 🎯 Цель workshop'а

Изучить создание мульти-агентных систем с использованием фреймворка **smolagents** от Hugging Face. За 30 минут мы создадим систему из 4 специализированных агентов, которые совместно найдут лучший курс обмена USD на рубли.

## 🏗️ Архитектура системы

Наша система состоит из 4 агентов:

1. **CurrencyAgent** 📊 - получает официальный курс USD/RUB от ЦБ РФ
2. **LocationAgent** 🗺️ - находит банки в радиусе 5 км от адреса пользователя  
3. **BankRatesAgent** 💰 - собирает курсы обмена валют в найденных банках
4. **DecisionAgent** 🧠 - анализирует данные и дает рекомендации

## 🚀 Быстрый старт

### Шаг 1: Установка зависимостей

```bash
# Клонируйте репозиторий или скачайте файлы
cd Agents_Workshop

# Установите зависимости
pip install -r requirements.txt
```

### Шаг 2: Настройка API ключей

1. Скопируйте файл с шаблоном:
```bash
cp .env.example .env
```

2. Заполните ваши API ключи в файле `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

**Где получить API ключи:**
- **OpenAI API ключ**: [platform.openai.com](https://platform.openai.com/api-keys)
- **Serper.dev ключ**: [serper.dev](https://serper.dev/) (бесплатный план: 2500 запросов/месяц)

### Шаг 3: Запуск workshop

```bash
jupyter notebook currency_exchange_workshop.ipynb
```

## 📁 Структура проекта

```
Agents_Workshop/
├── currency_exchange_workshop.ipynb  # Основной notebook для workshop
├── custom_tools.py                   # Кастомные инструменты для агентов
├── requirements.txt                  # Python зависимости
├── .env.example                      # Шаблон для API ключей
├── .env                              # Ваши API ключи (создается вами)
└── README.md                         # Этот файл
```

## 🛠️ Технологии

- **smolagents** - фреймворк для создания \"агентов, которые думают кодом\"
- **OpenAI API** - языковая модель для агентов
- **Serper.dev** - API для поиска в Google
- **ЦБ РФ API** - официальные курсы валют
- **Jupyter Notebooks** - интерактивная среда разработки

## 📚 План workshop'а (30 минут)

### 1. Введение (5 минут)
- Теория мульти-агентных систем
- Обзор smolagents фреймворка
- Архитектура нашего решения

### 2. Настройка окружения (5 минут)
- Установка библиотек
- Настройка API ключей
- Проверка подключений

### 3. Создание кастомных инструментов (10 минут)
- SerperSearchTool - поиск банков
- CBRCurrencyTool - официальные курсы
- BankRatesTool - банковские курсы
- UserInputTool - взаимодействие с пользователем

### 4. Создание агентов (7 минут)
- CurrencyAgent с официальными курсами
- LocationAgent для поиска банков
- BankRatesAgent для сбора курсов
- DecisionAgent для анализа и рекомендаций

### 5. Демонстрация (3 минуты)
- Запуск полного сценария
- Анализ результатов
- Обсуждение возможностей развития

## 🔧 Кастомные инструменты

### SerperSearchTool
- **Назначение**: Поиск банков через Google Search API
- **Входные данные**: поисковый запрос
- **Выходные данные**: список банков с адресами

### CBRCurrencyTool
- **Назначение**: Получение официального курса ЦБ РФ
- **API**: https://www.cbr-xml-daily.ru/daily_json.js
- **Выходные данные**: курс USD/RUB с датой

### BankRatesTool
- **Назначение**: Получение курсов обмена в банках
- **Особенность**: Использует демо-данные для workshop'а
- **Выходные данные**: курсы покупки/продажи USD

### UserInputTool
- **Назначение**: Интерактивное взаимодействие с пользователем
- **Использование**: Получение адреса пользователя

## 🎯 Ожидаемые результаты

После прохождения workshop'а вы научитесь:

✅ Создавать мульти-агентные системы с smolagents  
✅ Разрабатывать кастомные инструменты для агентов  
✅ Интегрировать внешние API в агентские системы  
✅ Координировать работу между специализированными агентами  
✅ Строить end-to-end решения с использованием ИИ  

## 🔍 Примеры использования

### Простой запуск одного агента:
```python
from smolagents import CodeAgent
from smolagents.models import OpenAIModel
from custom_tools import CBRCurrencyTool

model = OpenAIModel(model_name=\"gpt-3.5-turbo\")
agent = CodeAgent(tools=[CBRCurrencyTool()], model=model)
result = agent.run(\"Получи курс доллара от ЦБ РФ\")
print(result)
```

### Координация между агентами:
```python
# CurrencyAgent получает официальный курс
official_rate = currency_agent.run(\"Получи курс USD/RUB\")

# LocationAgent находит банки  
banks = location_agent.run(\"Найди банки рядом с адресом пользователя\")

# DecisionAgent анализирует и рекомендует
recommendation = decision_agent.run(f\"Проанализируй: {official_rate}, {banks}\")
```

## 🚧 Возможности расширения

### Краткосрочные улучшения:
- Добавить больше банков и источников данных
- Улучшить парсинг адресов и геолокацию
- Добавить кэширование для API запросов
- Создать веб-интерфейс

### Долгосрочное развитие:
- Мониторинг курсов в реальном времени
- Машинное обучение для предсказания курсов
- Интеграция с банковскими API
- Мобильное приложение
- Уведомления о выгодных курсах

## 📖 Дополнительные ресурсы

### Документация:
- [smolagents Documentation](https://huggingface.co/docs/smolagents/index)
- [Serper.dev API Docs](https://serper.dev/)
- [ЦБ РФ Web Services](https://www.cbr.ru/development/dws/)

### Полезные ссылки:
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Hugging Face Hub](https://huggingface.co/)
- [Multi-Agent Systems Theory](https://en.wikipedia.org/wiki/Multi-agent_system)

## 🤝 Поддержка и обратная связь

Если у вас возникли вопросы или предложения:

1. Проверьте раздел \"Troubleshooting\" ниже
2. Создайте issue в репозитории
3. Обратитесь к ведущему workshop'а

## 🐛 Troubleshooting

### Проблема: \"ModuleNotFoundError: No module named 'smolagents'\"
**Решение:**
```bash
pip install smolagents[toolkit]
```

### Проблема: \"Error: OPENAI_API_KEY not found\"
**Решение:**
1. Создайте файл `.env` из `.env.example`
2. Добавьте ваш OpenAI API ключ
3. Перезапустите notebook

### Проблема: \"Error making request to Serper API\"
**Решение:**
1. Проверьте правильность SERPER_API_KEY в файле `.env`
2. Убедитесь, что у вас есть доступ к интернету
3. Проверьте лимиты на вашем Serper аккаунте

### Проблема: Медленная работа агентов
**Решение:**
- Используйте более быструю модель (например, gpt-3.5-turbo вместо gpt-4)
- Упростите промпты агентов
- Добавьте кэширование API ответов

## 📝 Лицензия

Этот workshop создан в образовательных целях. Все используемые API имеют свои условия использования.

---

**Желаем успешного прохождения workshop'а! 🎉**

*Создано с использованием smolagents от Hugging Face*"