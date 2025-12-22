from flask import Flask
from flask import request, jsonify

import pandas as pd
import numpy as np
import pickle
import json
import time
from os.path import exists, getmtime

from src.reservoir_model import forecast_index
from src.index_grab import download_moex_data, data_preprocessing
from src.pymssa import MSSA

app = Flask(__name__)

def proximity(coord, obj): #Евклидова близость по координатам 
    lat, lon = coord
    duga=np.array([111.1, 62.8]) #градус дуги на широте Москвы
    pr=np.linalg.norm((obj[['lat', 'lon']].values - np.array(coord))*duga, axis=1)
    return pr.min(), pr.argmin()

@app.route("/price")
def price():
    resp='Okay'
    code=0
    modname='GBoost25_2c.pkl' #обучена на данных до 2025-06-30 с центром и парковками
    hor=float(request.args.get("horizon"))
    planchers_tot=int(request.args.get("planchers_tot"))
    height=float(request.args.get('height')) # высота обхекта (м)
    souterrain=int(request.args.get('souterrain')) # количество подземных этажей
    planchers_sur=int(request.args.get('planchers_sur')) # количество надземных этажей
    n_app=int(request.args.get('n_app')), # количество квартир
    superficie_tot=float(request.args.get('superficie_tot')) # общая площадь объекта 
    espace_de_vie=float(request.args.get('espace_de_vie')) # жилая площадь объекта
    lon=float(request.args.get('lon')) # долгота
    lat=float(request.args.get('lat')) # широта
    park=int(request.args.get('park')) # парковка
    horizon = [0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0]
    if hor not in horizon:
        code=-1
        hor=0
    hind=horizon.index(hor)
    data=[[planchers_tot,height,souterrain,planchers_sur,n_app,superficie_tot, espace_de_vie,lon,lat, park]]      
    uins=pd.DataFrame(columns=['planchers_tot','height','souterrain','planchers_sur','n_app','superficie_tot', 'espace_de_vie','lon','lat', 'park'],
                 data=data)   
    
    datadir='data/' #Место хранения справочников
    resdir='results/' #Место выгрузки результата xlsx
    
    #Близлежащие объекты
    metro=pd.read_csv(datadir+'metro.csv') # Станции метро
    rw=pd.read_csv(datadir+'railway.csv')  # Станции железной дороги
    med=pd.read_csv(datadir+'med.csv') # Поликлиники
    edu=pd.read_csv(datadir+'edu.csv') # Образоваьельные учреждения
    cem=pd.read_csv(datadir+'cem.csv') # Кладбища
    ind=pd.read_csv(datadir+'enterprises.csv') # Проьышленость в зонах
    uins['metroprox']=[proximity(x, metro)[0] for x in uins[['lat', 'lon']].values]
    uins['rwprox']=[proximity(x, rw)[0] for x in uins[['lat', 'lon']].values]
    uins['medprox']=[proximity(x, med[med.sort=='Городские поликлиники'])[0] for x in uins[['lat', 'lon']].values]
    uins['kidprox']=[proximity(x, edu[edu.sort=='ДОО'])[0] for x in uins[['lat', 'lon']].values]
    uins['schoolprox']=[proximity(x, edu[edu.sort=='Школы'])[0] for x in uins[['lat', 'lon']].values]
    uins['cemetprox']=[proximity(x, cem)[0] for x in uins[['lat', 'lon']].values]
    uins['industrprox']=[proximity(x, ind)[0] for x in uins[['lat', 'lon']].values]
    uins['liveratio']=uins['espace_de_vie']/uins['superficie_tot']
    uins['highratio']=uins['height']/uins['planchers_sur']
    uins['park']=[park]*len(uins)
    RSK=15.6/2/np.pi #радиус Садового кольца
    POKROVA_NA_RVU = pd.DataFrame({'lat':[55.7524], 'lon': [37.6231]}) #Координаты Покрова на рву - центр.
    uins['center']=[proximity(x, POKROVA_NA_RVU)[0] for x in uins[['lat', 'lon']].values]
    #uins['center']=np.where(uins['center'].values < RSK, 1, 0) #22.12.2025
    if (uins['highratio'].values[0]>5.)|(uins['highratio'].values[0]<2.):
        resp='Doubtfull number of storey (%i) for this height %.1f m.'%(planchers_sur, height)
        code=-2

    #Проверяем и обновляем базу индексов для прогноза:
    ifn=datadir+'dom_index.csv'
    if exists(ifn):
        ind_dat=getmtime(ifn)
        real_dat=time.time()
        if real_dat-ind_dat > 604800: #недела в секундах
            raw_data=download_moex_data(start_date='2016-12-28')
            processed_data = data_preprocessing(raw_data)
            if not processed_data.empty:
                processed_data.to_csv(ifn, index=False)
    else: 
        raw_data=download_moex_data(start_date='2016-12-28')
        processed_data = data_preprocessing(raw_data)
        if not processed_data.empty:
            processed_data.to_csv(ifn, index=False)
        else:
            resp='Can\'t predict price index. No data ' + ifn + ' Set horizon = 0.0'
            code=-3

    # прогнозируем индекс:
    if code == -3:
        index=np.ones(9)
    else:
        index=forecast_index(datadir+'dom_index.csv', 105)  
        index=index/index[0]
    if (index.max()>3)|(index.min()<.99):
        resp='Predicted price index looks doubtful: %.3f'%index
        code=-4

            
    # Официальная инфляциея по Распоряжению Департамента экономической политики и развития города Москвы № ДПР-Р-34.24 28.12.2024
    # Об утверждении прогнозных коэффициентов инфляции на 2025–2027 годы (с фактическими коэффициентами инфляции за период с 2023 по 2024 годы (по состоянию на 20.12.2024)) 
    infl_off=pd.read_csv(datadir+'inflatio.csv', parse_dates=[0]).set_index('date', drop=True)
    j=[]
    for h in horizon:
        i=pd.to_datetime(time.strftime('%Y-%m-%d', time.localtime()))+pd.Timedelta(f'{int(h*365)}D')
        j.append(str(i.year)+'-'+str(i.month).zfill(2)+'-01')
    today_index=infl_off.loc[j[0],'infl']
    official=infl_off.loc[j,'infl']/today_index

    #official=np.array([0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0])
    
    # Порядок факторов для модели:
    # ['количество этажей всего', 'Общая площадь объекта', 'lon', 'lat',
    #    'metroprox', 'rwprox', 'medprox', 'kidprox', 'schoolprox', 'cemetprox',
    #    'industrprox', 'liveratio', 'highratio', 'center',
    #    'Наличие подземной автостоянки']
    # Есть:
    #'planchers_tot','height','souterrain','planchers_sur','n_app','superficie_tot', 'espace_de_vie','lon','lat', 'park'
    Xy=uins[['planchers_tot', 'superficie_tot','lon', 'lat', 
             'metroprox', 'rwprox', 'medprox', 'kidprox', 'schoolprox', 'cemetprox', 'industrprox', 'liveratio', 'highratio', 
             'center','park']]         
    with open('src/'+modname, 'rb') as f:
        model = pickle.load(f) #Обученная модель
    y_pred=model.predict(Xy) #*index
    lat, lon = uins[['lat', 'lon']].values[0]
    price_pure=round(y_pred[0]*1000, 2)*today_index
    uins['center']=np.where(uins['center'].values < RSK, 1, 0) #22.12.2025
    result=pd.DataFrame({'Горизонт прогноза':horizon,
                         'Дата': j,
                         'Широта':[lat]*len(horizon),
                         'Долгота':[lon]*len(horizon), 
                         'количество этажей всего':[planchers_tot]*len(horizon),
                         'общая площадь объекта':[superficie_tot]*len(horizon),
                         'жилая площадь объекта':[espace_de_vie]*len(horizon),
                         'Индекс официальный':np.round(official, 3),
                         'Индекс Домклик':np.round(index,3),
                         'Цена за метр на текущую дату':[price_pure.round(3)]*len(horizon),
                         'Цена за метр по Домклик':np.round(price_pure*index,2),
                         'Цена за метр по официальной':np.round(price_pure*official,2),
                         'Внутри Садового': ['Да' if uins['center'].values[0] ==1 else 'Нет']*len(horizon),
                         'Наличие подземной автостоянки' : ['Да' if park ==1 else 'Нет']*len(horizon),
                         'Примечание':[resp]*len(horizon)
                        })

    fname=resdir+'prognose_'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    result.to_excel(fname+'.xlsx', index=False)
    result.to_csv(fname+'.csv', index=False)

    return jsonify({'response': resp,
                    'code': code,
                    'lat': lat,
                    'lon': lon,
                    'center': int(uins['center'].values[0]),
                    'pricemetr_today': price_pure.round(3),
                    'pricemetr_dom': list(np.round(price_pure*index,2)),
                    'pricemetr_off': list(np.round(price_pure*official,2)),
                    'price_index': list(np.round(index,3)),
                    'official':list(np.round(official,3)),
                    'forecast_horizon': horizon,
                    'pricemetr_future_dom': (price_pure*index[hind]).round(2),
                    'pricemetr_future_off': (price_pure*official[hind]).round(2),
                    'dates':j,
                    'parking':park,
                    #'f_year':i.year,
                    #'f_month':i.month,
                    'excel_name':fname+'.xlsx',
                    'csv_name':fname+'.csv'})

@app.route("/demand")
def demand():
    resp='Okay'
    code=0
    datadir='data/'
    #district = str(request.args.get("district"))
    lon=float(request.args.get('lon')) # долгота
    lat=float(request.args.get('lat')) # широта
    year=int(request.args.get("year"))
    month=int(request.args.get("month"))

    distr_coord=pd.read_csv(datadir+'distr_coord.csv')
    _, n=proximity((lat, lon), distr_coord)
    district=distr_coord.loc[n,'Район']
    
    deals=pd.read_csv(datadir+'d_amt25.csv', parse_dates=[0])
    #deals.drop(deals[deals['Дата регистрации']<='2021-07-01'].index, inplace=True)
    #deals.drop(deals[deals['Дата регистрации']>'2024-07-01'].index, inplace=True)
    deals['month']=pd.to_datetime(deals['Дата регистрации'].dt.year.astype(str)+'-'+deals['Дата регистрации'].dt.month.astype(str).str.zfill(2)+'-01')
    bins=[44, 65, 92]
    deals['class'] = np.digitize(deals['Площадь квартиры'], bins=bins, right=True)
    if district in deals.Район.unique():
        nsales=deals[deals.Район==district].groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        resp='Okay'
        code=0
    else:
        nsales=deals.groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        resp=f'District {district} is not in train data. Forecast on average'
        code=-1
    nsales=(nsales.T/nsales.sum(axis=1).values).T
    if len(nsales)<24:
        nsales=deals.groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        resp=f'Not enough train data for {district}. Forecast on average'
        code=-1
    nsales=(nsales.T/nsales.sum(axis=1).values).T    
    mssa = MSSA(n_components=12,
            window_size=12,
            verbose=True)
    mssa.fit(nsales)
    #t=time.localtime(time.time())
    #horizon=(pd.to_datetime(f'{year}-{month}-01')-pd.to_datetime(f'{t.tm_year}-{t.tm_mon}-01')).days//30
    horizon=(pd.to_datetime(f'{year}-{str(month).zfill(2)}-01')-deals.month.max()).days//30
    fc = mssa.forecast(horizon)
    fc=np.where(fc<0,  0., fc)
    result=(fc/fc.sum(axis=0))[:, -1].round(2)
    a={'district':district, 'date':f'{year}-{str(month).zfill(2)}-01', 'code': code, 'response':resp}
    a.update(dict(zip(['1','2','3','4'], result)))
    print(a)
    return jsonify(a)

if __name__=='__main__':
    app.run(debug=True) #, ssl_context='adhoc')





