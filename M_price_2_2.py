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
from pymssa import MSSA

app = Flask(__name__)

def proximity(coord, obj): #Евклидова близость по координатам 
    lat, lon = coord
    pr=np.linalg.norm(obj[['lat', 'lon']].values - np.array(coord), axis=1).min()
    return pr

@app.route("/price")
def price():
    resp='Okay'
    horizon = float(request.args.get("horizon"))
    planchers_tot=int(request.args.get("planchers_tot"))
    height=float(request.args.get('height')) # высота обхекта (м)
    souterrain=int(request.args.get('souterrain')) # количество подземных этажей
    planchers_sur=int(request.args.get('planchers_sur')) # количество надземных этажей
    n_app=int(request.args.get('n_app')), # количество квартир
    superficie_tot=float(request.args.get('superficie_tot')) # общая площадь объекта 
    espace_de_vie=float(request.args.get('espace_de_vie')) # жилая площадь объекта
    lon=float(request.args.get('lon')) # долгота
    lat=float(request.args.get('lat')) # широта

    if horizon not in [0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75, 2.0]:
        resp="Whong value for horizon %.2f, changed to 0.0"%horizon
        horizon=0.

    data=[[planchers_tot,height,souterrain,planchers_sur,n_app,superficie_tot, espace_de_vie,lon,lat]]                  
    uins=pd.DataFrame(columns=['planchers_tot','height','souterrain','planchers_sur','n_app','superficie_tot', 'espace_de_vie','lon','lat'],
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
    uins['metroprox']=[proximity(x, metro) for x in uins[['lat', 'lon']].values]
    uins['rwprox']=[proximity(x, rw) for x in uins[['lat', 'lon']].values]
    uins['medprox']=[proximity(x, med[med.sort=='Городские поликлиники']) for x in uins[['lat', 'lon']].values]
    uins['kidprox']=[proximity(x, edu[edu.sort=='ДОО']) for x in uins[['lat', 'lon']].values]
    uins['schoolprox']=[proximity(x, edu[edu.sort=='Школы']) for x in uins[['lat', 'lon']].values]
    uins['cemetprox']=[proximity(x, cem) for x in uins[['lat', 'lon']].values]
    uins['industrprox']=[proximity(x, ind) for x in uins[['lat', 'lon']].values]
    uins['liveratio']=uins['espace_de_vie']/uins['superficie_tot']
    uins['highratio']=uins['height']/uins['planchers_sur']
    
    if (uins['highratio'].values[0]>5.)|(uins['highratio'].values[0]<2.):
        resp='Doubtfull number of storey (%i) for this height %.1f m.'%(planchers_sur, height)
    
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
            horizon=0.

    # прогнозируем индекс:
    if  horizon>0.: #Горизонт 0 - в текущих ценах, index=1.
        index=forecast_index(datadir+'dom_index.csv', int(horizon*52)+1)
    else:
        index=1.
        
    if (index>3)|(index<.99):
        resp='Predicted price index looks doubtful: %.3f'%index

    #Набор фичей для модели уточняется, для прилагаемой он таков:
    #planchers_sur superficie_tot lon lat metroprox liveratio highratio
    Xy=uins.drop(['height','souterrain', 'n_app','espace_de_vie', 'planchers_tot', 
                  #'planchers_sur',
             #'liveratio', 'highratio',
             'industrprox','medprox', 'kidprox', 'schoolprox', 'cemetprox', 'rwprox',
           ], axis=1).dropna()             
    with open('src/GBoost.pkl', 'rb') as f:
        model = pickle.load(f) #Обученная модель
    y_pred=model.predict(Xy)*index
    lat, lon = uins[['lat', 'lon']].values[0]
    
    result=pd.DataFrame(data=[[lat, lon, planchers_tot, espace_de_vie, horizon, index, round(y_pred[0]*1000, 2), resp]], 
                        columns=['Широта', 'Долгота', 'количество этажей всего','жилая площадь объекта',
                                        'Горизонт прогноза','Индекс цен','Цена за метр (тыр)', 'Примечание'])
    fname=resdir+'prognose_'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())+'.xlsx'
    result.to_excel(fname, index=False)

    return jsonify({'response': resp,
                    'lat': lat,
                    'lon': lon,
                    'pricemetr': round(y_pred[0]*1000, 2),
                    'price_index': round(index, 3),
                    'forecast_horizon': horizon,
                    'file_name':fname })

@app.route("/demand")
def demand():
    resp='Okay'
    datadir='data/'
    district = str(request.args.get("district"))
    year=int(request.args.get("year"))
    month=int(request.args.get("month"))
    
    deals=pd.read_csv(datadir+'d_amt.csv', parse_dates=[0])
    #deals.drop(deals[deals['Дата регистрации']<='2021-07-01'].index, inplace=True)
    deals.drop(deals[deals['Дата регистрации']>'2024-07-01'].index, inplace=True)
    deals['month']=pd.to_datetime(deals['Дата регистрации'].dt.year.astype(str)+'-'+deals['Дата регистрации'].dt.month.astype(str).str.zfill(2)+'-01')
    bins=[44, 65, 92]
    deals['class'] = np.digitize(deals['Площадь квартиры'], bins=bins, right=True)
    if district in deals.Район.unique():
        nsales=deals[deals.Район==district].groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        respond='Okay'
    else:
        nsales=deals.groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        respond=f'District {district} is not in train data. Forecast on average'
    nsales=(nsales.T/nsales.sum(axis=1).values).T
    if len(nsales)<24:
        nsales=deals.groupby(['month', 'class'])['Дата регистрации'].count().unstack().fillna(0).rolling(3).mean().dropna()
        respond=f'Not enough train data for {district}. Forecast on average'
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
    a={'district':district, 'date':f'{year}-{str(month).zfill(2)}-01', 'respond':respond}
    a.update(dict(zip(['1','2','3','4'], result)))
    print(a)
    return jsonify(a)

if __name__=='__main__':
    app.run(debug=True) #, ssl_context='adhoc')
