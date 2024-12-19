import streamlit as st
import requests
from bs4 import BeautifulSoup
import pyttsx3
from gtts import gTTS
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import speech_recognition as sr

# Инициализация голосового синтезатора с проверкой доступных движков
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def main():
    # Выбор метода ввода
    choice = input("Выберите метод ввода (1 для текста, 2 для речи): ")

    if choice == '1':
        user_input = get_text_input()  # Ввод текста с клавиатуры
    elif choice == '2':
        user_input = recognize_speech()  # Голосовой ввод
    else:
        print("Неверный выбор.")
        return

    # Дальше используем введённый текст как обычно
    print(f"Вы ввели: {user_input}")
    # Здесь можно передать user_input в дальнейшую обработку, например, в модель для генерации ответа

# Функции для работы с речью
def text_to_speech(text):
    """Превращает текст в голос с использованием pyttsx3"""
    engine.say(text)
    engine.runAndWait()

def google_tts(text):
    """Превращает текст в речь с использованием Google Text-to-Speech (gTTS)"""
    tts = gTTS(text=text, lang='ru')
    tts.save("response.mp3")
    os.system("start response.mp3") 

# Локальное распознавание речи с использованием pocketsphinx
def recognize_speech():
    """Распознавание речи с использованием Google Web Speech API на русском языке"""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Скажите что-нибудь...")
        audio = recognizer.listen(source)

        try:
            # Используем Google Web Speech API для распознавания речи
            recognized_text = recognizer.recognize_google(audio, language="ru-RU")
            return recognized_text
        except sr.UnknownValueError:
            return "Не удалось распознать речь."
        except sr.RequestError:
            return "Ошибка подключения к интернету."

def get_text_input():
    """Текстовый ввод с клавиатуры"""
    print("Введите текст:")
    user_input = input()
    return user_input

# Функция для веб-скрейпинга
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

# Пример использования функции веб-скрейпинга
def generate_response(user_input):
    """Пример генерации ответа, который может использовать веб-скрейпинг"""
    if 'поиск' in user_input.lower():
        query = user_input.lower().replace('поиск', '').strip()
        result = web_scrape(query)
        return f"Я нашёл это: {result}"
    else:
        return "Это не поисковый запрос."

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

# Использование бесплатных решений для погоды и новостей (например, без API-ключей)
def get_online_data(query):
    try:
        if "погода" in query.lower():
            # Пример получения данных о погоде без использования API
            return random.choice([
                "Погода может быть непредсказуемой, но я постараюсь уточнить для тебя.",
                "Сейчас я посмотрю прогноз... дай мне секундочку!",
                "Хочешь знать, тепло ли сегодня? Я проверю!",
                "Похоже, сегодня лучше взять зонт. Но я уточню!",
                "Думаю, погода сегодня будет отличной для прогулки. Сейчас уточню."
            ])
        elif "новости" in query.lower():
            # Пример получения новостей без использования API
            return random.choice([
                "Мир не стоит на месте. Сейчас я найду что-то интересное для тебя.",
                "Давай посмотрим, что происходит вокруг. Новости уже на подходе!",
                "Я люблю узнавать новое. Сейчас поделюсь с тобой свежими новостями!",
                "Минуточку, посмотрю, что нового в мире.",
                "В новостях всегда что-то происходит. Хочешь узнать, что именно?"
            ])
        elif "краткое содержание" in query.lower():
            return random.choice([
                "Сейчас попробую всё собрать в пару предложений. Немного подожди.",
                "Я постараюсь сделать краткий пересказ, чтобы тебе было проще понять.",
                "Иногда сжатие текста — это искусство, но я попробую.",
                "Минуточку, сейчас я соберу самое важное для тебя.",
                "Попробую уложиться в несколько строк. Дай мне немного времени."
            ])
        elif "определение" in query.lower() or "метод" in query.lower() or "объяснение" in query.lower():
            return random.choice([
                "Хмм, кажется, это сложное слово. Сейчас попробую объяснить.",
                "У меня есть некоторые идеи. Позволь мне попробовать их сформулировать.",
                "Дай мне минутку, чтобы всё обдумать, и я постараюсь объяснить проще.",
                "Иногда определения могут быть немного путанными, но я попробую.",
                "Может, это звучит сложно, но я уверен, ты всё поймёшь."
            ])
        else:
            return random.choice([
                "Ммм, я постараюсь помочь... но если что-то пойдет не так, просто скажи мне, и я исправлю.",
                "Ну что ж, давай посмотрим... если что-то будет не так, просто не сердись, я попробую еще раз!"
            ])
        
    except Exception as e:
        return f"Ошибка получения данных: {str(e)}"

# Функция для получения данных о погоде
def get_weather(city_name):
    api_key = "ваш_ключ_API_для_погоды"  # Замените на свой ключ API
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric&lang=ru"
    response = requests.get(url)
    
    # Проверка успешности запроса
    if response.status_code == 200:
        data = response.json()
        weather = data["main"]
        temperature = weather["temp"]
        description = data["weather"][0]["description"]
        return f"Температура в {city_name}: {temperature}°C, {description}."
    else:
        return "Не удалось получить данные о погоде. Возможно, превышен лимит запросов."

# Обновлённая функция для анализа сообщения пользователя
def analyze_user_message(query):
    try:
        # Фразы для получения информации о погоде
        if "погода" in query.lower():
            # Извлекаем название города из запроса (например, "погода в Москве")
            match = re.search(r"погода в (\w+)", query.lower())
            if match:
                city_name = match.group(1)
                weather_info = get_weather(city_name)
                return weather_info
            else:
                return "Не могу распознать название города. Уточните, пожалуйста."

        # Фразы для поддержания разговора
        elif any(word in query.lower() for word in ["как ты", "что думаешь", "мнение", "интересно"]):
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

# Streamlit интерфейс
st.title("PersoComp Chatbot")

if user_input:
    response = analyze_user_message(user_input)
    if response:
        st.text_area("Ответ PersoComp:", value=response, height=100)
        text_to_speech(response)
    else:
        st.error("Не могу понять ваш запрос. Попробуйте спросить по-другому.")

# Использование метода ввода
if st.session_state.dialog_initialized:
    input_method = st.radio("Выберите метод ввода:", ('Текстовый ввод', 'Голосовой ввод'))

    # Обработка текстового ввода
    if input_method == 'Текстовый ввод':
        user_input = st.text_input("Введите ваш вопрос:", key="user_input")

    # Обработка голосового ввода
    elif input_method == 'Голосовой ввод':
        if st.button("Распознать речь"):
            user_input = recognize_speech()  # Запуск голосового ввода

    # Проверка и обработка текста
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


# Взаимодействие с пользователем через распознавание речи
if st.button("Распознать речь"):
    user_input = recognize_speech()  # Вызов функции для распознавания речи
    
    # Если речь распознана корректно
    if user_input not in ["Ошибка распознавания.", "Ошибка подключения к интернету."]:
        st.write(f"Распознано: {user_input}")
        response = generate_response(user_input)  # Генерация ответа на распознанный текст
        st.text_area("Ответ PersoComp:", value=response, height=100)
        text_to_speech(response)  # Преобразование текста в речь
    else:
        st.error(user_input)  # Вывод ошибки, если не удалось распознать речь


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
