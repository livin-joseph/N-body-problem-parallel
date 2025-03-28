from astroquery.jplhorizons import Horizons
import pandas as pd

id_list= {'Sun':10, 'Mercury':199, 'Venus' :299, 'Earth': 399, 'Mars':499, 'Jupiter' : 599 ,'Saturn' :699 ,'Uranus': 799,'Neptune': 899}

mass_data = {
    "Sun": 1.989e30,
    "Mercury": 3.301e23,
    "Venus": 4.868e24,
    "Earth": 5.972e24,
    "Mars": 6.417e23,
    "Jupiter": 1.898e27,
    "Saturn": 5.683e26,
    "Uranus": 8.681e25,
    "Neptune": 1.024e26
}

data=[]

for planet,id in id_list.items():
    obj = Horizons(
        id=str(id),          
        location='@sun',  
        epochs={'start': '2025-01-01', 'stop': '2025-01-02', 'step': '1d'},
    )
    vec = obj.vectors()
    df = vec.to_pandas()
    row=df.iloc[0][['x','y','z','vx','vy','vz']].values
    data.append([planet,mass_data[planet],*row])

col=['Name', 'Mass', 'x','y','z','vx','vy','vz']
planet_data=pd.DataFrame(data,columns=col)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("Planet Data:")
print(planet_data.to_string(index=False))
planet_data.to_csv("solar_system.csv", index=False)