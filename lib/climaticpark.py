# %%
# Importing necessary libraries (see configuration file climaticpark_env.yml in github repo)
#!pip install pybdshadow contextily folium pillow timezonefinder plotly

# %%
# Library Imports
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as dt 
from datetime import timedelta
import math
import os
from enum import Enum
import pytz
from scipy.integrate import solve_ivp


import branca
import folium
from folium.plugins import TimestampedGeoJson
import geopandas as gpd
from shapely.geometry import Polygon, mapping

from IPython.core.display import display
from IPython.display import IFrame

from suncalc import get_position
from pyproj import CRS,Transformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D,MaxPooling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# %%
class ShadowModule:

    """
    Module in charge of computing the shadows of the Target Parking Lot (TPL).

    Attributes:
        roofs (GeoDataFrame): Roofs structure of the TPL.
        spaces (GeoDataFrame): Parking spaces of the TPL
    """
        
    def __init__(self, roofs:gpd.GeoDataFrame, spaces:gpd.GeoDataFrame):
        self.roofs = roofs
        self.spaces= spaces
    
    
    def compute_coverage_rates(self, days_lst:list):
        """
        Compute coverage rates for each day in days_lst
        """
        coverage_rates_gdf = self.spaces.copy()# gpd.GeoDataFrame(geometry=self.spaces.geometry)
        coverage_rates_gdf = coverage_rates_gdf.set_crs(epsg=4326)
        
        local_tz = pytz.timezone('Europe/Madrid')

        shadows_lst = []
        for current_date in days_lst:
            # Compute shadow projections for roofs
            for hour in range(0,24):
                
                date_hour = dt.combine(current_date, datetime.time(hour, 0)) 
                date_hour = local_tz.localize(date_hour)  # Localizarlo en la zona horaria de Madrid

                coverage_rates= []
                try:
                    shadows_gdf = self._all_sunshadeshadow_sunlight(date_hour)
                    shadows_lst.append((date_hour, shadows_gdf))
                    

                    # Calculate coverage rates
                    intersection= coverage_rates_gdf.overlay(shadows_gdf, how='intersection')

                    
                    for index, parking_space in coverage_rates_gdf.iterrows():
                        space_total_area = parking_space.geometry.area
                        space_id = parking_space.space_id
                        space_shadow_area_lst= intersection.loc[intersection['space_id']==space_id, "geometry"]
                        space_shadow_area = space_shadow_area_lst.iloc[0].area if not space_shadow_area_lst.empty else 0
                        #space_shadow_area = intersection.loc[index, "geometry"].area if index in intersection.index else 0 #intersection.loc[index,"geometry"].area
                        space_coverage= space_shadow_area / space_total_area
                        coverage_rates.append(space_coverage)
                except Exception as e:
                    coverage_rates = [0] * len(coverage_rates_gdf)
                    print(f"WARN:: No coverage computed for date {date_hour}: {e}")


                """
                coverage_rates = []
                for index, parking_space in self.spaces.iterrows():
                    parking_space_gdf = gpd.GeoDataFrame(geometry=self..geometry)
                    parking_space_gdf = parking_space_gdf.set_crs(epsg=4326)
                    parking_space_gdf = parking_space_gdf.to_crs(epsg=shadows_gdf.crs.to_epsg())

                    intersection = gpd.overlay(parking_space_gdf, shadows_gdf, how='intersection')

                    intersection_area = intersection.geometry.area.sum()
                    parking_space_area = parking_space_gdf.geometry.area.sum()

                    coverage_rate = intersection_area / parking_space_area
                    coverage_rates.append(coverage_rate)
                """

                coverage_rates_gdf[f'coverage_rate_{date_hour.strftime("%Y-%m-%d %H:%M")}'] = coverage_rates

        self._coverage_rates = coverage_rates_gdf
        self._shadows_lst = shadows_lst
        coverage_rates_gdf.to_file("test.geojson", driver="GeoJSON")
        return coverage_rates_gdf

    def show_coverage_rate_map(self):
        # Extract the time columns (make sure they are sorted).
        coverage_cols = [col for col in self._coverage_rates.columns if "coverage_rate_" in col]
        coverage_cols.sort()

        # Create the base map centered on the area of the polygons.
        m = folium.Map(
            location=[self._coverage_rates.geometry.centroid.y.mean(), self._coverage_rates.geometry.centroid.x.mean()],
            zoom_start=16,
            max_zoom= 19,
            tiles="cartodb positron"
        )

        # Create a color palette based on the coverage values YlOrRd_09.
        colormap = branca.colormap.linear.Greys_09.scale(
            self._coverage_rates[coverage_cols].min().min(), self._coverage_rates[coverage_cols].max().max()
        )
        colormap.caption = "Coverage Rate"
        colormap.add_to(m)

        # Create a structure for TimestampedGeoJson.
        features = []

        for index, row in self._coverage_rates.iterrows():
            for time_index, col in enumerate(coverage_cols):
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),  # Convert to JSON
                    "properties": {
                        "time": col.replace("coverage_rate_", ""),  # Extract date and hour
                        "style": {
                            "fillColor": colormap(row[col]),  # Color based on coverage
                            "color": "black",
                            "weight": 0.5,
                            "fillOpacity": 0.7,
                        },
                        "popup": f"Coverage: {row[col]:.2f}"
                    }
                }
                features.append(feature)
        # Create `TimestampedGeoJson`
        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT1H",  # Time interval (1 hour)
            duration= "PT1H",
            add_last_point=False,
            auto_play=False,
            loop=True,
            max_speed=1,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
        ).add_to(m)
        
        folium.LayerControl().add_to(m)

        # Save map as HTML
        m.save(".climaticpark_coverage_rate_map.html")

        return m

    def show_shadow_map(self):

        # Customization
        bordersStyle = {
            'color': 'green',
            'weight': 0.5,
            'fillColor': 'blue',
            'fillOpacity': 0.4
        }

        # Create the base map centered on the area of the polygons.
        m = folium.Map(
            location=[self.spaces.geometry.centroid.y.mean(), self.spaces.geometry.centroid.x.mean()],
            zoom_start=16,
            max_zoom= 19,
            tiles="cartodb positron"
        )

        folium.GeoJson(self.spaces, name='Spaces', style_function=lambda x: bordersStyle).add_to(m)

        features = []

        for date, row in self._shadows_lst:
            for g in row.geometry:
                feature = {
                    "type": "Feature",
                    "geometry": mapping(g),  # Convert to JSON
                    "properties": {
                        "time": date.strftime("%Y-%m-%d %H:%M"),  
                        "style": {
                            "fillColor": 'grey',  
                            "color": "black",
                            "weight": 0.5,
                            "fillOpacity": 0.6,
                        },
                        "popup": "shadow"
                    }
                }
                features.append(feature)

        # Create `TimestampedGeoJson`
        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT1H",  # Time interval (1 hour)
            duration="PT1M",
            add_last_point=False,
            auto_play=False,
            loop=True,
            max_speed=1,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
        ).add_to(m)

        m.save(".climaticpark_roof_shadows_map.html")

        return m

    # Function to calculate shadow and sunlight for all rooftops
    def _all_sunshadeshadow_sunlight(self, date:datetime.datetime):
        roof_projected_df= self.roofs.copy()
        roof_projected_df['geometry'] = roof_projected_df.apply(lambda r: self._sunshadeshadow_sunlight(date, r[0]), axis=1)

        return roof_projected_df


    def _sunshadeshadow_sunlight(self, date:datetime.datetime, r:Polygon, sunshade_height=2):
        meanlon= r.centroid.y
        meanlat= r.centroid.x
        # obtain sun position
        sunPosition = get_position(date, meanlon, meanlat)
        if sunPosition['altitude'] < 0:
            raise ValueError("Given time before sunrise or after sunset")
            
        r_coords= np.array(r.exterior.coords)
        r_coords= r_coords.reshape(1,-1,2)
        shape = ShadowModule.lonlat2aeqd(r_coords,meanlon,meanlat)
        azimuth = sunPosition['azimuth']
        altitude = sunPosition['altitude']

        n = np.shape(shape)[0]
        distance = sunshade_height / math.tan(altitude)

        # calculate the offset of the projection position
        lonDistance = distance * math.sin(azimuth)
        latDistance = distance * math.cos(azimuth)

        shadowShape = np.zeros((1, 5, 2))
        shadowShape[:, :, :] += shape
        shadowShape[:, :, 0] = shape[:, :, 0] + lonDistance
        shadowShape[:, :, 1] = shape[:, :, 1] + latDistance
        shadowShape = ShadowModule.aeqd2lonlat(shadowShape,meanlon,meanlat)
        p = Polygon([[p[0], p[1]] for p in shadowShape[0]])
        return p
    
    @staticmethod
    def lonlat2aeqd(lonlat:np.ndarray, center_lon:float, center_lat:float):
        epsg = CRS.from_proj4("+proj=aeqd +lat_0="+str(center_lat) +
                            " +lon_0="+str(center_lon)+" +datum=WGS84")
        transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
        proj_coords = transformer.transform(lonlat[:, :, 0], lonlat[:, :, 1])
        proj_coords = np.array(proj_coords).transpose([1, 2, 0])
        return proj_coords
    
    @staticmethod
    def aeqd2lonlat(proj_coords:np.ndarray, meanlon:float, meanlat:float):
        epsg = CRS.from_proj4("+proj=aeqd +lat_0="+str(meanlat)+" +lon_0="+str(meanlon)+" +datum=WGS84")
        transformer = Transformer.from_crs( epsg,"EPSG:4326",always_xy = True)
        lonlat = transformer.transform(proj_coords[:,:,0], proj_coords[:,:,1])
        lonlat = np.array(lonlat).transpose([1,2,0])
        return lonlat



# %%
class DemandModule:

    """
    Module which simulates the drivers' behaviour of the Target Parking Lot (TPL) in terms o entry and exit hours

    Attributes:
        entry_exit_tupes (DataFrame): Historic entry and exit hours of the drivers in the TPL.
    """

    def __init__(self, entry_exit_tuples:pd.DataFrame):        
        self._entry_exit_tuples= entry_exit_tuples

    def train_demand_predictors(self, lr=0.9, sequence_length= 12, show_details=True, refresh_model=False):
        self._sequence_length= sequence_length

        self._gmm_models={}
        for hour, group in self._entry_exit_tuples.groupby('enter_hour'):
            X = group["exit_hour"].values  # Extract the feature data for this group.
            if X.shape[0]== 1:
                X= X.reshape(1, -1)
            else:
                X= X.reshape(-1,1)
            if X.shape[0]>=2:
                # Create and fit the GMM model
                gmm = GaussianMixture(n_components=1, random_state=42)
                gmm.fit(X)
                self._gmm_models[hour] = gmm

        model_path = os.path.join('_models', 'demand_cnnlstm_model.keras')

        n_incoming_veh_df= self._entry_exit_tuples.groupby('date').size().reset_index()
        n_incoming_veh_df['datetime'] = pd.to_datetime(n_incoming_veh_df['date'])

        # Define 'datetime' as index
        n_incoming_veh_df.set_index('datetime', inplace=True)

        # Rename 'num_vehicles' columns
        n_incoming_veh_df.rename(columns={0: "num_vehicles"}, inplace=True)

        # Delete original columns
        n_incoming_veh_df.drop(columns=["date"], inplace=True)

        self._scaler = MinMaxScaler()
        n_incoming_veh_df['num_vehicles'] = self._scaler.fit_transform(n_incoming_veh_df[['num_vehicles']])

        values = n_incoming_veh_df['num_vehicles'].values
        X, y = self._create_unidimensional_sequences(values, self._sequence_length)

        # Training-test data split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-lr, random_state=42)

        # Reshape data for CNN-LSTM model
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        self._last_sequence = X_test[-1].flatten() 

        if os.path.exists(model_path) and not refresh_model:
            print(f"\n\tLoading demand predictor from {model_path}...", end="")
            self._model = load_model(model_path)
        else: 
            print(f"\n\tTraining demand predictor...", end="")

            # CNN-LSTM composition
            self._model = Sequential([
                Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                LSTM(50, activation='relu', return_sequences=False),
                Dense(1)
            ])

            self._model.compile(optimizer='adam', loss='mse')

            _verbose= 0
            if show_details:
                _verbose= 1
            # EarlyStopping configuration
            early_stopping = EarlyStopping(
                monitor="val_loss",  # Target metric
                patience=10,         # Number of epochs to wait without improvement before stopping.
                restore_best_weights=True,  # Restore the best weights.Restaurar los mejores pesos
                verbose=_verbose          # Show mesages?
            )

            # Training with EarlyStopping
            self._model.fit(
                X_train,
                y_train,
                epochs=1000,
                batch_size=16,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],  # callback
                verbose=1
            )
            
            self._model.save(model_path)

        print("DONE!")

    # Create the input and output data for the time series.
    def _create_unidimensional_sequences(self, data, sequence_length:int):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def _predict_occupancy(self, n_days_ahead:int):

        future_predictions = []
        current_sequence = self._last_sequence.copy()

        for _ in range(n_days_ahead*24):
            # Reshape the current sequence to be compatible with the model.
            input_data = np.array(current_sequence).reshape((1, self._sequence_length, 1))
            
            # Predict the next value.
            next_pred = self._model.predict(input_data, verbose=0)[0][0]
            
            # Save the predicted value (denormalized).
            future_predictions.append(self._scaler.inverse_transform([[next_pred]])[0][0])

            # Update the input sequence with the new prediction.
            current_sequence = np.append(current_sequence[1:], next_pred)

        return future_predictions

   
    def generate_entry_exit_hours(self, date_lst:list):
        """
        Simulates vehicle parking occupancy and generates a table of occupancy information.
        """

        n_vehicles_per_hour = self._predict_occupancy(len(date_lst))
        simulated_occupancy={}
        hour= 0
        local_tz = pytz.timezone('Europe/Madrid')
        date_index= 0
        
        for n_vehicles in n_vehicles_per_hour: 
            current_date = date_lst[date_index]
            date_hour = dt.combine(current_date, datetime.time(hour, 0)) 
            date_hour = local_tz.localize(date_hour)  # Set it to the Madrid time zone.

            if hour in self._gmm_models:
                gm = self._gmm_models[hour]      
                new_samples, _ = gm.sample(n_samples=n_vehicles)
                simulated_occupancy[date_hour.strftime("%Y-%m-%d %H:%M")]=[math.floor(x[0]) for x in new_samples]
            hour = (hour+1) % 24
            if hour == 0:
                date_index += 1    
        return simulated_occupancy 

# %%
class AmbientModule:

    """
    Module that simulates the ambient temperature within the Target Parking Lot (TPL).

    Attributes:
        lat (float): Latitude of the TPL's spatial centroid.
        lon (float): Longitude of the TPL's spatial centroid.
    """
    
    def __init__(self, lat:float, lon:float):
        self.lat= lat
        self.lon= lon

    def train_ambient_temperature_model(self, start_date:datetime.datetime, end_date:datetime.datetime, lr=0.9, show_details=False, refresh_model=False):
        """
        Trains a LSTM model using the combined temperature data.

        :param combined_temp_data: DataFrame containing combined temperature data
        """

        ambient_temperaure_df= self._fetch_historical_temperature(start_date, end_date)
        model_path = os.path.join('_models', 'cabintemp_lstm_model.keras')

        # Data normalization
        scaler = MinMaxScaler(feature_range=(0, 1))

        data = ambient_temperaure_df.copy()
        data["temperature_2m"] = scaler.fit_transform(ambient_temperaure_df["temperature_2m"].values.reshape(-1, 1))

        look_back = 12  # Número de pasos anteriores a considerar
        X, y = ClimaticPark.create_sequences(data.values, look_back)

        # Train-test split
        train_size = int(len(X) * lr)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        self._last_sequence = X_test[-1].flatten() 
        
        model = None
        # Load or train model
        if os.path.exists(model_path) and not refresh_model:
            print(f"\n\tLoading ambient temperature predictor from {model_path}...", end="")
            model = load_model(model_path)
            print("DONE!")
        else:
            print(f"\n\tTraining and saving model in {model_path}...")

            _verbose= 0
            if show_details:
              _verbose= 1
            # EarlyStopping configuration
            early_stopping = EarlyStopping(
                monitor="val_loss",  # Métrica que se monitorea
                patience=10,         # Número de épocas de espera sin mejoras antes de detener
                restore_best_weights=True,  # Restaurar los mejores pesos
                verbose=_verbose           # Mostrar mensajes
            )


            # LSTM model composition
            model = Sequential()
            model.add(LSTM(50, input_shape=(look_back, 1)))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")

            # Training
            model.fit(
                X_train,
                y_train,
                epochs=1000,
                batch_size=16,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping], 
                verbose=1
            )
            self._last_sequence = X_test[-1].flatten() 

        model.save(model_path)

        self._model= model
        self._scaler= scaler

    def predict_ambient_temperature(self, days_lst:list):

        future_predictions = {}
        current_sequence = self._last_sequence.copy()

        local_tz = pytz.timezone('Europe/Madrid')

        for current_date in days_lst:
            for hour in range(0,24):
                date_hour = dt.combine(current_date, datetime.time(hour, 0)) 
                date_hour = local_tz.localize(date_hour)  

                date_hour_str= date_hour.strftime("%Y-%m-%d %H:%M")

                # Reshape the current sequence to be compatible with the model.
                input_data = np.array(current_sequence).reshape((1, len(current_sequence), 1))
                
                # Predict the next value.
                next_pred = self._model.predict(input_data, verbose=0)
                
                # Save the predicted value (denormalized).
                future_predictions[date_hour_str]=self._scaler.inverse_transform([[next_pred[0][0]]])[0][0]

                # Update the input sequence with the new prediction.
                current_sequence = np.append(current_sequence[1:], next_pred)

        return future_predictions

    # Function to fetch historical hourly data from the Open-Meteo API.
    def _fetch_historical_temperature(self, start_date:datetime.datetime, end_date:datetime.datetime):
          
        """
        Fetches historical hourly data from the Open-Meteo API.

        Args:
        - latitude (float): Latitude of the location.
        - longitude (float): Longitude of the location.
        - start_date (str): Start date in YYYY-MM-DD format.
        - end_date (str): End date in YYYY-MM-DD format.
        - parameters (list): Meteorological variables to query, e.g., ['temperature_2m', 'humidity_2m'].

        Returns:
        - pd.DataFrame: Hourly meteorological data as a DataFrame.
        """
        base_url = "https://archive-api.open-meteo.com/v1/archive"

        print(f"\n\tFetching wheather data from Open-Meteo...", end="")

        # Request's payload
        payload = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": 'temperature_2m',
            "timezone": "auto"
        }

        # API's get call
        response = requests.get(base_url, params=payload)

        if response.status_code == 200:
            # Convert the JSON response to a DataFrame.
            data = response.json()
            if "hourly" in data:
                df = pd.DataFrame(data["hourly"])
                df['time'] = pd.to_datetime(df['time'])
                df= df.set_index('time')
                print("DONE!")
                return df
            else:
                print("No data found in the response.")
                return pd.DataFrame()
        else:
            print(f"Request error: {response.status_code} - {response.text}")
            return pd.DataFrame()
          
    """
    def _forecast_uncovered_cabin_temperatures(self, ambient_temp):
        ambient_temp_scaled = self.temp_scaler.fit_transform(ambient_temp)
        y_pred = self.cabin_temp_model.predict(ambient_temp_scaled)
        y_pred_rescaled = self.cabin_temp_scaler.inverse_transform(y_pred)
        return y_pred_rescaled"
    """

# %%
class Vehicle:

    """
    Class that represents a simulated vehicle in the TPL.

    Attributes:
        vehicle_id (str): Unique identifier of the vehicle.
        entry_timestamp (datetime): Date and hour when the venicle entry the TPL.
        exit_timestamp (datetime): Date and hour when the vehicle exit the TPL.
        assigned_space_id (int): Identifier of the space in the TPL where the vehicle is located.
        initial_cabin_temp (float): Vehicle's cabin temperature when it enters the TPL:
        target_cabin_temp (float): Confort vehicle's cabin temperature that the driver wants to.
        coolingPower (float): Cooling power of the vehicle's air conditioning system.
        cabinVolume (float): Volume of the vehicle's cabin in m^2
        airDensity (float): Contextual air density.
        specificHeatAir (float): Contextual heat air.
    """

    def __init__(self,
                 vehicle_id:str, 
                 entry_timestamp:datetime.datetime, 
                 exit_timestamp:datetime.datetime, 
                 assigned_space_id:int, 
                 initial_cabin_temp=25, 
                 target_cabin_temp= 23,
                 coolingPower=5000,
                 cabinVolume=3, 
                 airDensity=1.2, 
                 specificHeatAir=1005):
        
        self.vehicle_id = vehicle_id
        self.entry_timestamp = entry_timestamp
        self.exit_timestamp = exit_timestamp
        self.assigned_space_id = assigned_space_id
        self.initial_cabin_temp= initial_cabin_temp #celsius
        self.final_cabin_temp= initial_cabin_temp
        self.target_cabin_temp = target_cabin_temp
        self.coolingPower = coolingPower
        self.cabinVolume = cabinVolume
        self.airDensity = airDensity
        self.specificHeatAir = specificHeatAir

    def compute_energy_consumption(self):
        solution = self._simulate_cooling()
        self.cooling_time = solution.t_events[0][0] if solution.t_events[0].size > 0 else solution.t[-1]
        self.fuel_consumption = self._fuel_consumption(self.cooling_time)

    def _cooling_dynamics(self, t, T_cabin):
        dTdt = -self.coolingPower / (self.airDensity * self.cabinVolume * self.specificHeatAir)
        return dTdt

    def _target_temperature_reached(self, t, T_cabin:list):
        return T_cabin[0] - self.target_cabin_temp
    
    _target_temperature_reached.terminal = True  # Event attribute

    def _simulate_cooling(self, max_time=3600):
        # As `solve_ivp` requires the event function to be independent, we define a wrapper
        def event(t, T_cabin): return self._target_temperature_reached(t, T_cabin)
        event.terminal = True

        solution = solve_ivp(
            self._cooling_dynamics, [0, max_time], [self.final_cabin_temp],
            events=event, dense_output=True
        )
        return solution

    def _fuel_consumption(self, cooling_time:float):
        # Convert energy in joules to liters (assuming 1 liter = 3.6e6 J)
        return (self.coolingPower * cooling_time) / 3.6e6


    def to_dict(self):
        return {'Vehicle id.': int(self.vehicle_id), "Entry": self.entry_timestamp.strftime('%Y-%m-%d %H:%M'), "Exit": self.exit_timestamp.strftime('%Y-%m-%d %H:%M'), "Space Id.": int(self.assigned_space_id)}

    def __str__(self):
        return f"Vehicle({self.vehicle_id}, entry:{self.entry_timestamp.strftime('%Y-%m-%d %H:%M')}, exit:{self.exit_timestamp.strftime('%Y-%m-%d %H:%M')}, space:{self.assigned_space_id}, init_temp:{self.initial_cabin_temp}, final_temp:{self.final_cabin_temp})"

    def __repr__(self):
        return self.__str__()

class OccupancyModule:

    def __init__(self, spaces:gpd.GeoDataFrame, gates:gpd.GeoDataFrame):
        self._spaces= spaces
        self._gates= gates
        
        self._spaces[['nearest_gate_id', 'nearest_gate_dist']] = self._spaces.geometry.apply(lambda poly: self._nearest_point(poly))
        
    def _nearest_point(self, polygon:Polygon):
        distances = self._gates.geometry.distance(polygon)
        nearest_idx = distances.idxmin()
        return pd.Series([self._gates.loc[nearest_idx, 'id'], distances[nearest_idx]])
    
    def _select_weighted_random_row(self, gdf:gpd.GeoDataFrame, weight_col='nearest_gate_dist'):

        try:
            if weight_col not in gdf.columns:
                raise ValueError(f"Column {weight_col} does not exist in GeoDataFrame")
            
            weights = gdf[weight_col].values

            max_finite = np.nanmax(weights[weights != np.inf])  # Maximum finite value in p
            if np.isfinite(max_finite):
                weights = np.where(np.isinf(weights), max_finite, weights)  # Replace inf with the maximum finite value.
            
            min_finite = np.nanmin(weights[weights != 0])  # Maximum finite value in p

            weights = weights + min_finite
            inv_weights = 1 / weights    
            norm_inv_weights = inv_weights / np.sum(inv_weights)
            selected_idx = np.random.choice(gdf.index, p=norm_inv_weights)
            return selected_idx
        except:
            print("WARN: It was not possible to assign space to incoming vehicle")
            return -1

    def simulate_occupancies(self, days_lst:list, entry_exits:dict):

        local_tz = pytz.timezone('Europe/Madrid')

        simulated_occupancy = self._spaces.copy()

        last_column_name = None
        occupied_spaces_dict = {}
        vehicles_dict= {}
        current_vehicle_id = 0
        for current_date in days_lst:
            for entry_hour in range(0,24):
                
                date_hour = dt.combine(current_date, datetime.time(entry_hour, 0)) 
                date_hour = local_tz.localize(date_hour)  

                date_hour_str= date_hour.strftime("%Y-%m-%d %H:%M")

                current_column_name= f'occupancy_{date_hour_str}'

                if last_column_name is not None:
                    simulated_occupancy[current_column_name]= simulated_occupancy[last_column_name]
                else:
                    simulated_occupancy[current_column_name]= -1 #False

                new_vacant_spaces_lst= []
                if date_hour_str in occupied_spaces_dict:
                    new_vacant_spaces_lst= occupied_spaces_dict[date_hour_str]
                
                for vacant_space in new_vacant_spaces_lst:
                    simulated_occupancy.at[vacant_space,current_column_name]= -1 #False

                if date_hour_str in entry_exits:
                    exit_hours_lst = entry_exits[date_hour_str]
                    for exit_hour in exit_hours_lst:
                        selected_space = self._select_weighted_random_row(simulated_occupancy[simulated_occupancy[current_column_name]==-1])
                        if selected_space >= 0:

                            simulated_occupancy.at[selected_space,current_column_name]= current_vehicle_id#True

                            if exit_hour > entry_hour:
                                date_exit_hour = dt.combine(current_date, datetime.time(exit_hour, 0)) 
                            elif exit_hour == entry_hour:
                                date_exit_hour = dt.combine(current_date, datetime.time((exit_hour+1)%24, 0)) 
                            else:
                                next_date= current_date + timedelta(days=1)
                                date_exit_hour = dt.combine(next_date, datetime.time(exit_hour, 0)) 

                            date_exit_hour = local_tz.localize(date_exit_hour)  
                            date_exit_hour_str= date_exit_hour.strftime("%Y-%m-%d %H:%M")

                            vehicle = Vehicle(current_vehicle_id, date_hour, date_exit_hour, selected_space)
                            vehicles_dict[current_vehicle_id]= vehicle
                            current_vehicle_id += 1

                            exit_hour_spaces = occupied_spaces_dict.get(date_exit_hour_str,[])
                            exit_hour_spaces.append(selected_space)
                            occupied_spaces_dict[date_exit_hour_str]= exit_hour_spaces

                last_column_name= current_column_name
        self._simulated_occupancy = simulated_occupancy
        self._vehicles_dict= vehicles_dict

        return vehicles_dict
    
    def show_occupancy_map(self):
        # Extract the time columns (make sure they are sorted).
        occupancy_cols = [col for col in self._simulated_occupancy.columns if "occupancy_" in col]
        occupancy_cols.sort()

        # Create the base map centered on the area of the polygons
        m = folium.Map(
            location=[self._simulated_occupancy.geometry.centroid.y.mean(), self._simulated_occupancy.geometry.centroid.x.mean()],
            zoom_start=16,
            max_zoom= 19,
            tiles="cartodb positron"
        )

        def get_color(cell_value):
            if cell_value >= 0:
                return "#FF0000"
            return  "#CCCCCC"
        
        def get_popup_value(cell_value):
            if cell_value in self._vehicles_dict:
                return str(self._vehicles_dict[cell_value].to_dict())
            return "Empty"

        # Create a structure for TimestampedGeoJson
        features = []

        for index, row in self._simulated_occupancy.iterrows():
            for time_index, col in enumerate(occupancy_cols):
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),  # Convert to JSON
                    "properties": {
                        "time": col.replace("occupancy_", ""),  # Extract date and hour
                        "style": {
                            "fillColor": get_color(row[col]),
                            "color": "black",
                            "weight": 0.5,
                            "fillOpacity": 0.7,
                        },
                        "popup": get_popup_value(row[col])  
                    }
                }
                features.append(feature)
        # Crate `TimestampedGeoJson`
        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT1H",  # Time interval (1 hoir)
            duration= "PT1H",
            add_last_point=False,
            auto_play=False,
            loop=True,
            max_speed=1,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
        ).add_to(m)
        
        folium.LayerControl().add_to(m)

        # Save map as HTML file
        m.save(".climaticpark_occupancy_map.html")

        return m        


# %%
class CabinTemperatureModule:

    """
    Module in charge of computing the cabin temperature of the vehicles under certain circumstances.

    Attributes:
        data_no_roof (DataFrame): Dataset of vehicles' cabin temperature when there is no roof.
        data_roof (DataFrame): Dataset of vehicles' cabin temperature when the are protected from the sun exposure by a roof.
    """

    def __init__(self, data_no_roof:pd.DataFrame, data_roof:pd.DataFrame):
        
        self._data_no_roof= data_no_roof
        self._data_roof= data_roof
        self._temp_data = pd.concat([self._data_no_roof, self._data_roof], axis=0)

        X= self._temp_data['T temp_ext coverage'.split()].values
        y= self._temp_data['temp_int'].values

        self._cabin_temp_pred = LinearRegression().fit(X, y)

    def simulate_cabin_temperatures(self, vehicles_dict:dict, coverage_rates_gdf:gpd.GeoDataFrame, forecasted_ambient_temp_dict:dict):

        self._cabin_temp_gdf= coverage_rates_gdf.copy()
        self._cabin_temp_gdf= self._cabin_temp_gdf.drop(columns='height nearest_gate_id nearest_gate_dist'.split())
        self._cabin_temp_gdf.columns= [c.replace('coverage_rate_', 'cabin_temp_') for c in self._cabin_temp_gdf.columns]

        for c in self._cabin_temp_gdf.columns:
            if c.startswith('cabin_temp_'):
                self._cabin_temp_gdf[c]= '-'
    
        for vehicle_id, vehicle in vehicles_dict.items():
            space_id= vehicle.assigned_space_id
            entry_date = vehicle.entry_timestamp
            current_date = vehicle.entry_timestamp
            date_lst= []
            while current_date <= vehicle.exit_timestamp:
                date_lst.append(current_date)
                current_date += timedelta(hours=1)

            for i in range(len(date_lst)):
                d = date_lst[i]
                d_str= d.strftime("%Y-%m-%d %H:%M")
                amb_temp= forecasted_ambient_temp_dict[d_str]
                coverage_rate = coverage_rates_gdf.at[space_id, f'coverage_rate_{d_str}']
                
                cabin_temp = self._cabin_temp_pred.predict(np.array([i+1,amb_temp,coverage_rate]).reshape(1,-1))
                
                self._cabin_temp_gdf.at[space_id, f'cabin_temp_{d_str}']= cabin_temp[0]

            vehicle.final_cabin_temp= cabin_temp[0]
        return vehicles_dict
    
    def show_cabintemp_map(self):

        cabintemp_cols = [col for col in self._cabin_temp_gdf.columns if "cabin_temp_" in col]
        cabintemp_cols.sort()

        m = folium.Map(
            location=[self._cabin_temp_gdf.geometry.centroid.y.mean(), self._cabin_temp_gdf.geometry.centroid.x.mean()],
            zoom_start=16,
            max_zoom= 19,
            tiles="cartodb positron"
        )

        colormap = branca.colormap.linear.YlOrRd_04.scale(
            self._cabin_temp_gdf.replace('-', np.nan)[cabintemp_cols].min().min(), self._cabin_temp_gdf.replace('-', np.nan)[cabintemp_cols].max().max()
        )
        colormap.caption = "Coverage Rate"
        colormap.add_to(m)

        features = []

        for index, row in self._cabin_temp_gdf.iterrows():
            for time_index, col in enumerate(cabintemp_cols):
                
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),  # Convert to JSON
                    "properties": {
                        "time": col.replace("cabin_temp_", ""),  # Extract date and hoir
                        "style": {
                            "fillColor": colormap(row[col]) if row[col] != '-' else colormap(0),  # Coverage color
                            "color": "black",
                            "weight": 0.5,
                            "fillOpacity": 0.7,
                        },
                        "popup": f"Cabin temp:{row[col]:.2f} Cº" if row[col] != '-' else '-'
                    }
                }
                features.append(feature)

        # Create  `TimestampedGeoJson`
        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT1H",  # Time interval (1 hour)
            duration= "PT1H",
            add_last_point=False,
            auto_play=False,
            loop=True,
            max_speed=1,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm",
        ).add_to(m)
        
        folium.LayerControl().add_to(m)

        # Save map as HTML file
        m.save(".climaticpark_cabintemp_map.html")

        return m

# %%
class ClimaticParkState(Enum):

    "Enumeration defining the states of the simulation"

    INIT = 1
    READY = 2
    LAUNCHED = 3

# %%
class ClimaticPark:

    """
    Core module of the library in charge of orchestrating all the other modules. It lauches and controls the overall simulation
    Attributes:
        file_name_lots (str): Path to the geojson file comprising the spatial distribution of the TPL spaces
        file_name_roofs (str): Path to the geojson file comprising the spatial data of the roofs installed in the TPL.
        file_name_coords (str): Path to the csv file comprising the spatial centroid of the TPL.
        file_name_gates (str): Path to the csv file comprising the lat-lot coordinates with the location of the entry and exit gates of the TPL.
        file_name_cabintem (str): Path to the csv file comprising the dataset of cabin temperatures of the vehicles.
    """

    def __init__(self, 
                 file_name_lots:str,
                 file_name_roofs:str,
                 file_name_coords:str,
                 file_name_gates:str,
                 file_name_cabintem:str):
        """
        Initializes the ClimaticPark object by loading all necessary files.
        """
        # Load GeoJSON files for lots and roofs
        self.lots_data = ClimaticPark.load_geojson(file_name_lots)
        self.lots_data['space_id']= list(range(len(self.lots_data )))
        self.roofs_data = ClimaticPark.load_geojson(file_name_roofs)

        # Add 'height' column to roofs_data with a value of 1 for all rows
        if (self.lots_data is not None) and ('height' not in self.lots_data.columns):
            self.lots_data['height'] = 0 # Assuming a default height of 1

        # Add 'height' column to roofs_data with a value of 1 for all rows
        if (self.roofs_data is not None) and ('height' not in self.roofs_data.columns):
            self.roofs_data['height'] = 2  # Assuming a default height of 1

        # Load CSV files for coordinates, historical data, and additional data
        self.coords_data = ClimaticPark.load_csv(file_name_coords)
        self.gates_data = ClimaticPark.load_csv(file_name_gates)

        # Convert to GeoDataFrame
        self.gates_data = gpd.GeoDataFrame(self.gates_data, geometry=gpd.points_from_xy(self.gates_data.longitude, 
                                                                                        self.gates_data.latitude), 
                                                                                        crs="EPSG:4326")  # WGS 84
        self.gates_data['id']= self.gates_data.index
      
        data_no_roof =  ClimaticPark.load_csv(os.path.join('data', 'cabin_temperature_no_roof.csv'))
        data_no_roof['coverage']=0
        data_roof =  ClimaticPark.load_csv(os.path.join('data', 'cabin_temperature_w_roof.csv'))
        data_roof['coverage']=1

        self.recorded_cabin_temp = pd.read_csv(file_name_cabintem, index_col=0)
        # Cast DateTime column to datetime type
        self.recorded_cabin_temp['DateTime'] = pd.to_datetime(self.recorded_cabin_temp['DateTime'])
        # Set DateTime column as index
        self.recorded_cabin_temp.set_index('DateTime', inplace=True)
        # Resample the data to a 1-hour frequency (calculating the mean).
        self.recorded_cabin_temp = self.recorded_cabin_temp.resample('h').mean()

        self.entry_exit_tuples= pd.read_csv(os.path.join('data', 'entry_exit_tuples_clean.csv'), index_col=0, dtype={'id_subject':str}, parse_dates=['date'])
        
        os.makedirs('_models', exist_ok=True)
        self.cabin_temp_model = None
        self.cabin_coverage_model= None
        print("Generating Demand Module...", end="")
        self.demand_module = DemandModule(self.entry_exit_tuples)
        print("DONE!")

        print("Generating Ocupancy Module...", end="")
        self.occupancy_module = OccupancyModule(self.lots_data, self.gates_data)
        print("DONE!")
        
        lat = self.coords_data['latitude'].iloc[0]  
        lon = self.coords_data['longitude'].iloc[0]
        
        print("Generating Shadow Module...", end="")
        self.shadow_module = ShadowModule(self.roofs_data, self.lots_data)
        print("DONE!")

        print("Generating Ambient Module...", end="")
        self.ambient_module = AmbientModule(lat,lon)
        print("DONE!")

        print("Generating Cabin Temperature Module...", end="")
        self.cabin_module = CabinTemperatureModule(data_no_roof, data_roof)
        print("DONE!")

        self._state = ClimaticParkState.INIT

    @staticmethod
    def load_geojson(file_name:str):
        """
        Loads a GeoJSON file into a GeoDataFrame.
        """
        if file_name:
            return gpd.read_file(file_name).set_geometry("geometry")
        else:
            print(f"No GeoJSON file provided for {file_name}.")
            return None

    @staticmethod
    def load_csv(file_name:str):
        """
        Loads a CSV file into a DataFrame.
        """
        if file_name:
            return pd.read_csv(file_name)
        else:
            print(f"No CSV file provided for {file_name}.")
            return None

    def prepare_simulation(self, lr=0.8, display_details=False):
        print("Preparing simulation for TPL...")

        init_date =  self.recorded_cabin_temp.index[0].date()
        final_date =  self.recorded_cabin_temp.index[-1].date()

        # Process temperature data
        print("\tTraining ambient temperature predictors...",end="")
        self.ambient_module.train_ambient_temperature_model(init_date, final_date)
        print("DONE!")

        print("\tTraining demand predictors...",end="")
        self.demand_module.train_demand_predictors()  

        self._state = ClimaticParkState.READY

        print("Simulation ready to go!!")

    def launch_simulation(self, n_days_ahead:int, display_details=True):
        """
        Lauch simulation for n_days_ahead 
        """
        print("Starting simulation of TPL...")

        
        init_day = self.entry_exit_tuples['date'].max()
        
        madrid_tz = pytz.timezone("Europe/Madrid")
        init_day = madrid_tz.localize(init_day)

        date_lst = [init_day + timedelta(days=i) for i in range(n_days_ahead)]
        entry_exits= self.demand_module.generate_entry_exit_hours(date_lst)

        simulated_vehicles_dict = self.occupancy_module.simulate_occupancies(date_lst, entry_exits)    
        simulated_coverage_rates= self.shadow_module.compute_coverage_rates(date_lst)

        forecasted_ambient_temp_dict= self.ambient_module.predict_ambient_temperature(date_lst)

        self.vehicles_dict = self.cabin_module.simulate_cabin_temperatures(simulated_vehicles_dict, simulated_coverage_rates, forecasted_ambient_temp_dict)
 
        self._state = ClimaticParkState.LAUNCHED
        print("Simulation launched. You can get access to the simulation data.")


    def show_coverage_rates(self):
        if self._state == ClimaticParkState.LAUNCHED:
            return self.shadow_module.show_coverage_rate_map()
        else:
            print("You must first launch the simulation by calling the launch_simulation method.")
    
    def show_roofs_projected_shadows(self):
        if self._state == ClimaticParkState.LAUNCHED:
            return self.shadow_module.show_shadow_map()
        else:
            print("You must first launch the simulation by calling the launch_simulation method.")

    def show_occupancy(self):
        if self._state == ClimaticParkState.LAUNCHED:
            return self.occupancy_module.show_occupancy_map()
        else:
            print("You must first launch the simulation by calling the launch_simulation method.")

    def show_cabin_temps(self):
        if self._state == ClimaticParkState.LAUNCHED:
            return self.cabin_module.show_cabintemp_map()
        else:
            print("You must first launch the simulation by calling the launch_simulation method.")

    def compute_energy_consumption(self):

        if self._state == ClimaticParkState.LAUNCHED:
            vehicle_lst = []
            for v_id, v in self.vehicles_dict.items():
                v.compute_energy_consumption()
                vehicle_lst.append(v.__dict__)
            
            vehicles_df = pd.DataFrame(vehicle_lst)
            return vehicles_df
        else:
            print("You must first launch the simulation by calling the launch_simulation method.")
        
   
    # Prepare the dataset for sequences.
    @staticmethod
    def create_sequences(data, look_back:int):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back]) 
            y.append(data[i + look_back])  
        return np.array(X), np.array(y)

