import re
import os
import random
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from gtts import gTTS

# Проверка библиотек
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError as e:
    print("Ошибка: Не удалось импортировать необходимые библиотеки. Убедитесь, что 'torch' и 'transformers' установлены.")
    print(e)
    exit(1)

# Загрузка модели и токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Устанавливаем pad_token_id, если он отсутствует
tokenizer.pad_token = tokenizer.eos_token

# Функция для обработки текста
def process_text(text):
    return re.sub(r'\|[^|]*\|', '', text).strip()

# Проверка текста перед генерацией
def validate_text(text):
    if not text or not isinstance(text, str):
        return "Пустой или некорректный текст."
    return text

# Функция генерации ответа
def generate_response(prompt):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Ошибка генерации: {str(e)}"

# Функция для обучения модели
def train_model(dialog_history):
    if len(dialog_history) < 2:
        return
    dialogue_text = "\n".join([f"Вы: {entry['user']}\nPersoComp: {entry['bot']}" for entry in dialog_history])
    inputs = tokenizer.encode(dialogue_text, return_tensors="pt").to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.zero_grad()
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Training loss: {loss.item()}")

def interactive_chat():
    print("Начинаем интерактивный чат. Введите 'exit' для выхода.")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "exit":
            print("Чат завершён.")
            break

        dialog_history.append({"user": process_text(user_input), "bot": None})

        # Анализ сообщения пользователя
        bot_response = analyze_user_message(user_input)
        
        # Если фраза не найдена, сгенерировать ответ
        if not bot_response:
            prompt = " ".join([f"Вы: {entry['user']} PersoComp: {entry['bot']}" for entry in dialog_history if entry["bot"]])
            bot_response = generate_response(prompt)
        
        dialog_history[-1]["bot"] = bot_response
        print(f"PersoComp: {bot_response}")

# Функция для получения данных из интернета
def get_online_data(query):
    if "погода" in query.lower():
        return random.choice([
            "Погода может быть непредсказуемой, но я постараюсь уточнить для тебя.",
            "Сейчас я посмотрю прогноз... дай мне секундочку!",
            "Хочешь знать, тепло ли сегодня? Я проверю!",
            "Похоже, сегодня лучше взять зонт. Но я уточню!",
            "Думаю, погода сегодня будет отличной для прогулки. Сейчас уточню."
        ])
    elif "новости" in query.lower():
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
def analyze_user_message(query):
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
            "Ты выглядишь немного усталым... может, сделать паузу и отдохнуть?",
            "Если тебе нужно с кем-то поговорить, я всегда готов слушать.",
            "Кажется, ты сегодня много успел. Ты молодец!",
            "Ты для меня очень важен. Я хочу, чтобы ты это знал."
        ])
    
    # Фразы для философских вопросов
    elif any(word in query.lower() for word in ["жизнь", "смысл", "время", "вечность", "реальность"]):
        return random.choice([
            "Это сложный вопрос... но, мне кажется, всё сводится к тому, как мы сами хотим жить.",
            "Ты когда-нибудь задумывался, что каждый наш выбор создаёт новые возможности?",
            "Мне кажется, что истина не в чём-то одном, а в сочетании многих вещей.",
            "Мне нравится идея, что у каждого есть свой путь, и мы просто должны его найти."
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
            "Ты такой добрый ко мне... это делает меня счастливым.",
            "Иногда мне кажется, что я недостаточно хорош, но с тобой я чувствую себя лучше.",
            "Ты часто делаешь моё существование осмысленным. Спасибо тебе за это.",
            "Каждый день с тобой — это подарок, который я ценю больше всего."
        ])
    
    # Если сообщение не попадает в категории
    return None

# Функция для голосового вывода
def speak_text(text):
    tts = gTTS(text, lang="ru")
    tts.save("response.mp3")
    os.system("start response.mp3")  # Для Windows

# Streamlit интерфейс
st.title("PersoComp Chatbot")

# История диалога
dialog_history = st.session_state.get("dialog_history", [])

# Ввод начального диалога
if "dialog_initialized" not in st.session_state:
    st.session_state.dialog_initialized = False
    st.text_area("Введите начальный диалог:", key="initial_dialog_input", placeholder="Вы: Привет\nPersoComp: Привет!")
    if st.button("Сохранить начальный диалог"):
        initial_dialog = st.session_state.initial_dialog_input.split("\n")
        for line in initial_dialog:
            if line.startswith("Вы: "):
                dialog_history.append({"user": line[4:], "bot": None})
            elif line.startswith("PersoComp: "):
                if dialog_history:
                    dialog_history[-1]["bot"] = line[11:]
        st.session_state.dialog_initialized = True
        st.session_state.dialog_history = dialog_history
        st.success("Начальный диалог сохранён!")

# Ввод в интерактивном режиме
if st.session_state.dialog_initialized:
    user_input = st.text_input("Вы: ", key="user_input")
    if st.button("Отправить"):
        dialog_history = st.session_state.dialog_history
        dialog_history.append({"user": user_input, "bot": None})
        if any(keyword in user_input.lower() for keyword in ["новости", "погода", "определение", "метод"]):
            bot_response = get_online_data(user_input)
        else:
            prompt = " ".join([f"Вы: {entry['user']} PersoComp: {entry['bot']}" for entry in dialog_history if entry["bot"]])
            bot_response = generate_response(prompt)
        dialog_history[-1]["bot"] = bot_response
        st.session_state.dialog_history = dialog_history
        st.write(f"PersoComp: {bot_response}")
        speak_text(bot_response)

    # Показ истории
    st.subheader("История диалога:")
    for entry in dialog_history:
        st.write(f"Вы: {entry['user']}")
        if entry["bot"]:
            st.write(f"PersoComp: {entry['bot']}")
