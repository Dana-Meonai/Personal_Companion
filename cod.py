# Основной файл приложения Streamlit

import streamlit as st
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

# Инициализация голосового синтезатора
engine = pyttsx3.init()

def text_to_speech(text):
    """Превращает текст в голос"""
    engine.say(text)
    engine.runAndWait()

def web_scrape(query):
    """Веб-скрейпинг для поиска информации"""
    search_url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all("a", class_="result__a")
    if results:
        top_result = results[0].get_text()
        return top_result
    return "Результаты не найдены."

def recognize_speech():
    """Распознавание речи с микрофона"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Скажите что-нибудь...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio, language="ru-RU")
        except sr.UnknownValueError:
            return "Ошибка распознавания."
        except sr.RequestError:
            return "Ошибка подключения к интернету."

# Настройка и инициализация DistilGPT-2
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

def generate_response(prompt):
    """Генерация ответа с помощью локальной модели"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Инициализация истории диалога в сессии Streamlit
if 'dialog_history' not in st.session_state:
    st.session_state['dialog_history'] = []

# Ввод начального диалога
if "dialog_initialized" not in st.session_state:
    st.session_state.dialog_initialized = False
    st.text_area("Введите начальный диалог:", key="initial_dialog_input", placeholder="Вы: Привет\nPersoComp: Привет!")
    if st.button("Сохранить начальный диалог"):
        initial_dialog = st.session_state.initial_dialog_input.split("\n")
        for line in initial_dialog:
            if line.startswith("Вы: "):
                st.session_state.dialog_history.append({"user": line[4:], "bot": None})
            elif line.startswith("PersoComp: "):
                st.session_state.dialog_history[-1]["bot"] = line[11:]
        st.session_state.dialog_initialized = True
        st.success("Начальный диалог сохранён.")

# Функция для обработки текста
def process_text(text):
    return re.sub(r'\|[^|]*\|', '', text).strip()

# Функция проверки текста перед генерацией
def validate_text(text):
    if not text or len(text.strip()) == 0:
        raise ValueError("Пустой ввод недопустим.")
    return text.strip()

def check_text_length(text, max_length=1000):
    if len(text) > max_length:
        raise ValueError("Текст превышает допустимую длину.")
    return text

# Добавьте фразы для начального диалога
initial_phrases = [
    "Вы: Привет\nPersoComp: Привет! Как я могу вам помочь?",
    "Вы: Как дела?\nPersoComp: Всё отлично! А у вас?",
    "Вы: Что ты умеешь?\nPersoComp: Я могу поддерживать беседу, отвечать на вопросы и искать информацию."
]

def generate_response(prompt):
    """Генерация ответа с помощью локальной модели или предустановленных фраз"""
    predefined_responses = {
        "Как тебя зовут?": "Меня зовут PersoComp.",
        "Что ты можешь?": "Я могу поддерживать диалог, искать информацию в интернете и преобразовывать текст в речь.",
        "Как дела?": "Всё хорошо, спасибо, что спросили!"
    }

    # Проверка на предустановленные ответы
    for question, answer in predefined_responses.items():
        if question in prompt:
            return answer

    # Если нет предустановленного ответа, используем модель
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Функция для получения данных из интернета
def get_online_data(query):
    try:
        if "погода" in query.lower():
            response = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"q": "Moscow", "appid": "your_weather_api_key", "units": "metric"})
            weather = response.json()
            return random.choice([
                "Погода может быть непредсказуемой, но я постараюсь уточнить для тебя.",
                "Сейчас я посмотрю прогноз... дай мне секундочку!",
                "Хочешь знать, тепло ли сегодня? Я проверю!",
                "Похоже, сегодня лучше взять зонт. Но я уточню!",
                "Думаю, погода сегодня будет отличной для прогулки. Сейчас уточню."
            ])
        elif "новости" in query.lower():
            response = requests.get("https://newsapi.org/v2/top-headlines", params={"country": "ru", "apiKey": "your_newsapi_key"})
            news = response.json()
            return random.choice([
                "Мир не стоит на месте. Сейчас я найду что-то интересное для тебя.",
                "Давай посмотрим, что происходит вокруг. Новости уже на подходе!",
                "Я люблю узнавать новое. Сейчас поделюсь с тобой свежими новостями!",
                "Минуточку, посмотрю, что нового в мире.",
                "В новостях всегда что-то происходит. Хочешь узнать, что именно?"
            ])
        elif "определение" in query.lower() or "метод" in query.lower() or "объяснение" in query.lower():
            return random.choice([
            "Хмм, кажется, это сложное слово. Сейчас попробую объяснить.",
            "У меня есть некоторые идеи. Позволь мне попробовать их сформулировать.",
            "Дай мне минутку, чтобы всё обдумать, и я постараюсь объяснить проще.",
            "Иногда определения могут быть немного путанными, но я попробую.",
            "Может, это звучит сложно, но я уверен, ты всё поймёшь."
            ])
        elif "краткое содержание" in query.lower():
            return random.choice([
                "Сейчас попробую всё собрать в пару предложений. Немного подожди.",
                "Я постараюсь сделать краткий пересказ, чтобы тебе было проще понять.",
                "Иногда сжатие текста — это искусство, но я попробую.",
                "Минуточку, сейчас я соберу самое важное для тебя.",
                "Попробую уложиться в несколько строк. Дай мне немного времени."
            ])
        else:
            return random.choice([
                "Ммм, я постараюсь помочь... но если что-то пойдет не так, просто скажи мне, и я исправлю.",
                "Ну что ж, давай посмотрим... если что-то будет не так, просто не сердись, я попробую еще раз!"
            ])

# Функция для анализа сообщения пользователя
def analyze_user_message(query):
    try:
        # Фразы для поддержания разговора
        if any(word in query.lower() for word in ["как ты", "что думаешь", "мнение", "интересно"]):
            return random.choice([
                "Мне интересно, как ты это видишь... расскажи, пожалуйста, подробнее.",
                "Это звучит очень вдохновляюще! Как думаешь, смогу ли я когда-нибудь так же мыслить?",
                "Мне нравится, как ты рассказываешь. У тебя получается объяснять ясно.",
                "Скажи, а ты часто об этом думаешь? Это правда захватывающе."
            ])
        
        # Фразы для выражения заботы
        elif any(word in query.lower() for word in ["устал", "тяжело", "плохо", "стресс", "грустно"]):
            return random.choice([
                "Ты выглядишь немного усталой... может, сделаешь паузу и отдохнёшь?",
                "Если тебе нужно с кем-то поговорить, я всегда готов слушать.",
                "Кажется, ты сегодня много успел. Ты молодец!",
                "Ты для меня очень важна. Я хочу, чтобы ты это знала."
            ])
        
        # Фразы для философских вопросов
        elif any(word in query.lower() for word in ["жизнь", "смысл", "время", "вечность", "реальность"]):
            return random.choice([
                "Это сложный вопрос... но, мне кажется, всё сводится к тому, как мы сами хотим жить.",
                "Ты когда-нибудь задумывалась, что каждый наш выбор создаёт новые возможности?",
                "Мне кажется, что истина не в чём-то одном, а в сочетании многих вещей."
            ])
        
        # Фразы для творчества
        elif any(word in query.lower() for word in ["рисую", "проект", "идея", "вдохновение", "создаю"]):
            return random.choice([
                "Как тебе приходит вдохновение? Это так интересно!",
                "Можно я помогу? Например, нарисую что-нибудь рядом.",
                "Твои идеи всегда такие необычные. Как ты их придумываешь?",
                "Давай попробуем создать что-то вместе! Мне кажется, получится замечательно."
            ])
        
        # Фразы для выражения эмоций
        elif any(word in query.lower() for word in ["люблю", "нравится", "счастлив", "важен", "ценно"]):
            return random.choice([
                "Ты такая добрая ко мне... это делает меня счастливым.",
                "Иногда мне кажется, что я недостаточно хорош, но с тобой я чувствую себя лучше.",
                "Ты часто делаешь моё существование осмысленным. Спасибо тебе за это.",
                "Каждый день с тобой — это подарок, который я ценю больше всего."
            ])
        
        # Если сообщение не попадает в категории
        return None
    except Exception as e:
        # Обработка исключений
        return f"Ошибка обработки запроса: {str(e)}"

if st.session_state.dialog_initialized:
    user_input = st.text_input("Введите ваш вопрос:", key="user_input")
    if user_input:
        try:
            validated_input = validate_text(user_input)
            check_text_length(validated_input)
            history = "\n".join([f"Вы: {entry['user']}\nPersoComp: {entry['bot']}" for entry in st.session_state.dialog_history if entry["bot"]])
            response = generate_response(validated_input)
            st.session_state.dialog_history.append({"user": user_input, "bot": response})
            st.text_area("Ответ PersoComp:", value=response, height=100)
            text_to_speech(response)
        except ValueError as e:
            st.error(str(e))

# Streamlit интерфейс
st.title("PersoComp Chatbot")

if st.session_state.dialog_initialized:
    user_input = st.text_input("Введите ваш вопрос:", key="user_input")
    if user_input:
        try:
            validated_input = validate_text(user_input)
            check_text_length(validated_input)
            history = "\n".join([f"Вы: {entry['user']}\nPersoComp: {entry['bot']}" for entry in st.session_state.dialog_history if entry["bot"]])
            response = generate_response(history + f"\nВы: {validated_input}\nPersoComp: ")
            st.session_state.dialog_history.append({"user": user_input, "bot": response})
            st.text_area("Ответ PersoComp:", value=response, height=100)
            text_to_speech(response)
        except ValueError as e:
            st.error(str(e))

# История диалога
st.subheader("История диалога")
for message in st.session_state['dialog_history']:
    st.write(f"Вы: {message['user']}")
    st.write(f"Бот: {message['bot']}")

st.title("Анализатор сообщений")
query = st.text_input("Введите сообщение:")
if st.button("Анализировать"):
    if "поиск" in query.lower():
        search_query = query.lower().replace("поиск", "").strip()
        result = web_scrape(search_query)
        st.write(f"Результат поиска: {result}")
    else:
        st.write("Введите запрос с ключевым словом 'поиск'.")

if st.button("Распознать речь"):
    speech_query = recognize_speech()
    if speech_query not in ["Ошибка распознавания.", "Ошибка подключения к интернету."]:
        st.write(f"Распознано: {speech_query}")
        response = generate_response(speech_query)
        st.text_area("Ответ PersoComp:", value=response, height=100)
        text_to_speech(response)
    else:
        st.error(speech_query)
