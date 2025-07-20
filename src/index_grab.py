# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:09:25 2025

@author: Роман Высоцкий
"""
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import time

def download_moex_data(start_date='2016-12-28', end_date=None, chunk_size=100):
    """
    Загружает исторические данные индекса MREDC с MOEX
    
    Параметры:
        start_date (str): Начальная дата в формате 'YYYY-MM-DD'
        end_date (str): Конечная дата (по умолчанию - текущая дата)
        chunk_size (int): Количество дней загружаемых за один запрос (макс. 100)
    
    Возвращает:
        pd.DataFrame: Объединенный DataFrame со всеми данными
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    base_url = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/MREDC.csv"
    params = {
        'iss.only': 'history',
        'iss.dp': 'comma',
        'iss.df': '%Y-%m-%d',
        'iss.tf': '%H:%M:%S',
        'iss.dtf': '%Y.%m.%d %H:%M:%S',
        'from': start_date,
        'limit': chunk_size,
        'start': 0,
        'sort_order': 'TRADEDATE',
        'sort_order_desc': 'asc'
    }
    
    all_data = []
    current_start = start_date
    
    while True:
        params['from'] = current_start
        params['till'] = end_date
        
        print(f"Загружаю данные с {current_start} по {end_date}...")
        
        response = requests.get(base_url, params=params)
        response.encoding = 'cp1251'
        
        if response.status_code != 200:
            raise Exception(f"Ошибка загрузки данных: {response.status_code}")
        
        # Пропускаем первую строку с метаданными
        data = pd.read_csv(StringIO(response.text), sep=';', header=1, decimal=',')
        
        if data.empty:
            break
            
        all_data.append(data)
        
        # Получаем последнюю дату в загруженных данных
        last_date = pd.to_datetime(data['TRADEDATE']).max()
        current_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Проверяем, достигли ли мы конечной даты
        if pd.to_datetime(current_start) > pd.to_datetime(end_date):
            break
            
        # Задержка между запросами, чтобы не перегружать сервер
        # time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data).reset_index(drop=True)

def data_preprocessing(raw_data):
    """
    Обработка загруженных данных
    
    Параметры:
        raw_data (pd.DataFrame): Сырые данные из MOEX
        
    Возвращает:
        pd.DataFrame: Обработанные данные
    """
    if raw_data.empty:
        return pd.DataFrame()
    
    data = raw_data.copy()
    
    # Выбираем нужные колонки и преобразуем типы
    data = data[['TRADEDATE', 'CLOSE']]
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'], dayfirst=True)
    
    # Сортируем по дате
    data = data.sort_values('TRADEDATE').reset_index(drop=True)
    
    # Добавляем логарифмическую доходность
    data['log_return_close'] = np.log(data['CLOSE'].rolling(13).mean()).diff()
    
    return data

# Пример использования:
if __name__ == "__main__":
    # Загружаем данные (можно указать конкретную конечную дату)
    raw_data = download_moex_data(start_date='2016-12-28', end_date='2025-03-19')
    
    # Обрабатываем данные
    processed_data = data_preprocessing(raw_data)
    
    # Сохраняем результат
    if not processed_data.empty:
        processed_data.to_csv('mredc_index_data.csv', index=False)
        print(f"Данные успешно сохранены. Всего записей: {len(processed_data)}")
    else:
        print("Не удалось загрузить данные")