import os
import requests
import json
import xmltodict
from typing import Any, Dict, List, Tuple
from smolagents.tools import Tool


class SerperSearchTool(Tool):
    """Инструмент для поиска банков через Serper.dev API"""
    name = "serper_search"
    description = "Searches for information using Serper.dev Google Search API. Useful for finding banks near a location."
    inputs = {"query": {"type": "string", "description": "Search query to execute"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: SERPER_API_KEY not found in environment variables"
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'q': query,
            'num': 10
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            results = response.json()
            
            # Форматируем результаты
            formatted_results = []
            if 'organic' in results:
                for result in results['organic'][:5]:  # Берем первые 5 результатов
                    formatted_results.append(f"Название: {result.get('title', 'N/A')}\n"
                                           f"Адрес: {result.get('snippet', 'N/A')}\n"
                                           f"Ссылка: {result.get('link', 'N/A')}\n")
            
            return "\n".join(formatted_results) if formatted_results else "Банки не найдены"
        
        except requests.RequestException as e:
            return f"Error making request to Serper API: {str(e)}"
        except Exception as e:
            return f"Error processing Serper API response: {str(e)}"


class CBRCurrencyTool(Tool):
    """Инструмент для получения официального курса валют от ЦБ РФ"""
    name = "cbr_currency"
    description = "Gets official currency rates from Central Bank of Russia (CBR)"
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        url = "https://www.cbr-xml-daily.ru/daily_json.js"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Получаем курс USD
            usd_rate = data['Valute']['USD']['Value']
            date = data['Date']
            
            return f"Официальный курс ЦБ РФ на {date[:10]}:\nUSD/RUB: {usd_rate:.4f} руб."
        
        except requests.RequestException as e:
            return f"Error fetching CBR data: {str(e)}"
        except Exception as e:
            return f"Error processing CBR data: {str(e)}"


class UserInputTool(Tool):
    """Инструмент для получения ввода от пользователя"""
    name = "user_input"
    description = "Gets input from the user interactively"
    inputs = {"prompt": {"type": "string", "description": "Prompt to show to the user"}}
    output_type = "string"

    def forward(self, prompt: str) -> str:
        try:
            user_input = input(f"{prompt}: ")
            return user_input.strip()
        except Exception as e:
            return f"Error getting user input: {str(e)}"

    

class AddressToCoordsTool(Tool):
    '''Инструмент для перевода адреса в координаты'''
    name = 'geocoder'
    description = 'Geocoder which get cooridinates by address'
    inputs = {"address": {"type": "string", "description": "Address of the bank"}}
    output_type = "string"

    def forward(self, address: str) -> str:
        lat, lon = 0.0, 0.0
        api_key = os.environ["GEOAPIFY_API_KEY"]
        try:
            response = requests.get(
                "https://api.geoapify.com/v1/geocode/search",
                params={"text": address, "lang": "ru", "limit": 1, "apiKey": api_key},
                timeout=30,
            )
            response.raise_for_status()

            lat = response.json()['features'][0]['properties']['lat']
            lon = response.json()['features'][0]['properties']['lon']
        except:
            print("No coordinates")

        return str((lat, lon))
