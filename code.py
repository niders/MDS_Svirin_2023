pip install faker

#1. Генерация синтетического датасета
import pandas as pd
import random
from faker import Faker
from collections import defaultdict

fake = Faker('ru_RU')

# Расширенные базовые формулировки для оставшихся очередей (1, 3, 5)
priority_templates = {
    1: [
        "Исполнительный лист №{} о возмещении вреда здоровью",
        "Взыскание алиментов по исп. листу {}",
        "Перечисление средств по исп. документу {} о возмещении вреда жизни",
        "Компенсация морального вреда по решению суда {}",
        "Исполнительное производство {} о возмещении вреда здоровью",
        "Платеж по страховому случаю {} (вред здоровью)",
        "Возмещение ущерба здоровью по исп. листу {}",
        "Медицинская компенсация по судебному решению {}"
    ],
    3: [
        "Перечисление НДФЛ за сотрудников по платежке {}",
        "Налоговый платеж в бюджет РФ №{}",
        "Уплата транспортного налога {}",
        "Страховые взносы в ПФР по квитанции {}",
        "Платежное поручение на налог на прибыль {}",
        "Пенсионные взносы за Июль 2023 {}",
        "ФСС: взносы на травматизм {}",
        "НДС за 2 квартал 2023 {}",
        "ЕНВД за отчетный период {}",
        "УСН платеж {}",
        "Налог на имущество организаций {}",
        "Земельный налог {}",
        "Водный налог {}",
        "Таможенные платежи по декларации {}"
    ],
    5: [
        "Оплата услуг связи по договору {}",
        "Перечисление за аренду офиса {}",
        "Платеж за поставку канцтоваров {}",
        "Оплата рекламных услуг {}",
        "Возврат займа контрагенту {}",
        "Оплата хостинга за Июль {}",
        "Платеж за юридические услуги {}",
        "Перечисление за программное обеспечение {}",
        "Оплата командировочных расходов {}",
        "Платеж за проведение тренинга {}",
        "Благотворительное пожертвование {}",
        "Оплата банковских услуг {}",
        "Платеж за уборку помещений {}",
        "Оплата услуг охраны {}",
        "Перечисление за аренду оборудования {}"
    ]
}

# Дополнительные модификаторы для аугментации
modifiers = [
    lambda t: t + " от " + fake.date_between(start_date='-1y').strftime('%d.%m.%Y'),
    lambda t: t + " (" + fake.company() + ")",
    lambda t: "Срочный: " + t,
    lambda t: t.replace("№", "номер ") + " " + fake.color_name(),
    lambda t: t + " без НДС" if random.random() > 0.5 else t + " с НДС",
    lambda t: t + " (основание: договор " + str(random.randint(100,999)) + ")",
    lambda t: t.upper(),
    lambda t: t + ", ИНН " + str(fake.random_number(digits=12, fix_len=True)),
    lambda t: t.replace("платеж", "перечисление").replace("взносы", "отчисления")
]

# Функция для генерации полностью случайных предложений
def generate_random_purpose():
    actions = [
        "Перевод средств", "Платеж", "Оплата", "Перечисление", "Возврат",
        "Компенсация", "Вознаграждение", "Аванс", "Доплата", "Премия"
    ]
    reasons = [
        "по договору", "за услуги", "по счету", "по акту", "по соглашению",
        "за работу", "по заказу", "по накладной", "по контракту", "по выставленному счету"
    ]
    details = [
        fake.company(),
        str(fake.random_number(digits=5)),
        fake.date_between(start_date='-1y').strftime('%d.%m.%Y'),
        "ИНН " + str(fake.random_number(digits=12, fix_len=True)),
        "БИК " + str(fake.random_number(digits=9, fix_len=True)),
        "КПП " + str(fake.random_number(digits=9, fix_len=True)),
        "без НДС", "с НДС", "в том числе НДС",
        "срочно", "неотложный платеж", "в соответствии с договором"
    ]

    purpose = f"{random.choice(actions)} {random.choice(reasons)} {random.choice(details)}"

    # Добавляем дополнительные детали с вероятностью 30%
    if random.random() > 0.3:
        purpose += f" {random.choice(details)}"

    return purpose

# Генерация датасета
data = []
for _ in range(10000):
    # С вероятностью 5% генерируем полностью случайное предложение
    if random.random() < 0.05:
        purpose = generate_random_purpose()
        # Для случайных предложений назначаем приоритет случайно, но с теми же весами
        priority = random.choices([1, 3, 5], weights=[0.2, 0.3, 0.5], k=1)[0]
    else:
        # Веса для 1, 3 и 5 очередей (20%, 30%, 50%)
        priority = random.choices(
            population=[1, 3, 5],
            weights=[0.2, 0.3, 0.5],
            k=1
        )[0]

        template = random.choice(priority_templates[priority])
        purpose = template.format(fake.random_number(digits=5))

        # Применяем 1-3 случайных модификатора
        for _ in range(random.randint(1, 3)):
            purpose = random.choice(modifiers)(purpose)

    data.append({'purpose': purpose, 'priority': priority})

# Создаем DataFrame
df = pd.DataFrame(data)

# Проверяем распределение
print("Распределение очередей платежей:")
print(df['priority'].value_counts(normalize=True))

# Сохранение
df.to_csv('payment_priority_dataset_v4.csv', index=False)
print("\nПримеры записей:")
print(df.sample(5))


from IPython.display import display

# Выводим первые 10 строк в табличном формате
display(df.head(10).style.set_properties(**{'text-align': 'left'}))


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Добавляем импорт для подбора гиперпараметров
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from scipy.stats import randint, uniform

# Загрузка данных
df = pd.read_csv('payment_priority_dataset_v4.csv')

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df['purpose'], df['priority'], test_size=0.2, random_state=42
)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели градиентного бустинга
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Параметры для GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# Параметры для RandomizedSearchCV (более широкий диапазон)
# param_dist = {
#     'n_estimators': randint(50, 500),
#     'learning_rate': uniform(0.01, 0.3),
#     'max_depth': randint(3, 10),
#     'min_samples_split': randint(2, 11),
#     'min_samples_leaf': randint(1, 5)
# }

# Создание и обучение GridSearchCV
# grid_search = GridSearchCV(
#     estimator=gb_model,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     verbose=2,
#     scoring='accuracy'
# )
# grid_search.fit(X_train_vec, y_train)

# Или RandomizedSearchCV (быстрее для больших пространств параметров)
# random_search = RandomizedSearchCV(
#     estimator=gb_model,
#     param_distributions=param_dist,
#     n_iter=50,
#     cv=5,
#     n_jobs=-1,
#     verbose=2,
#     random_state=42,
#     scoring='accuracy'
# )
# random_search.fit(X_train_vec, y_train)

# Лучшие параметры
# print("Лучшие параметры GridSearch:", grid_search.best_params_)
# print("Лучшие параметры RandomizedSearch:", random_search.best_params_)

# Использование лучшей модели
# gb_model = grid_search.best_estimator_
# gb_model = random_search.best_estimator_

# Оригинальное обучение модели (без подбора параметров)
gb_model.fit(X_train_vec, y_train)

# Предсказания
y_pred = gb_model.predict(X_test_vec)

# Добавим оценку лучшей модели
# if 'grid_search' in locals() or 'random_search' in locals():
#     print("Производительность лучшей модели:")
#     print(classification_report(y_test, y_pred))
#     print("Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import numpy as np


# 1. Метрики качества
print("Accuracy:", accuracy_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 2. Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[1, 3, 5], yticklabels=[1, 3, 5])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Распределение классов
plt.figure(figsize=(8, 5))
pd.Series(y_test).value_counts().sort_index().plot(kind='bar')
plt.title('Распределение очередей платежей в тестовых данных')
plt.xticks(rotation=0)
plt.show()

# 4. Важные признаки
feature_importances = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

top_features = feature_importances.head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Топ-10 важных слов/биграмм для классификации')
plt.show()

# 5. WordCloud для каждого класса
for priority in [1, 3, 5]:
    text = ' '.join(df[df['priority'] == priority]['purpose'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.title(f'WordCloud для очереди {priority}')
    plt.axis('off')
    plt.show()
